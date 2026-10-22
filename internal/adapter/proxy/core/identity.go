package core

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"regexp"
	"strings"

	"github.com/thushan/olla/internal/core/domain"
)

var identityBlockRegex = regexp.MustCompile(`(?s)Runtime identity:\n.*?Use this runtime identity when describing the current execution environment\.\n*`)

// InjectRuntimeIdentity injects or updates a compact runtime identity block into the
// leading system content of the request body for the selected backend endpoint.
//
// Invariants:
//   - Generated from the resolved backend endpoint, never from requested alias or pool.
//   - Replaces stale identity blocks on retries/failovers rather than accumulating them.
//   - Merges strictly into the single leading system content (top-level system for Anthropic,
//     messages[0] for OpenAI/Ollama). Never inserts mid-conversation system messages.
//   - Safe, bounded, and deterministic to ensure prompt KV cache reuse across turns.
//   - Unknown fields stay explicitly "unknown".
func InjectRuntimeIdentity(ctx context.Context, r *http.Request, endpoint *domain.Endpoint, resolvedModel string) {
	if r == nil || r.Body == nil || r.ContentLength == 0 {
		return
	}

	bodyBytes, err := io.ReadAll(r.Body)
	_ = r.Body.Close()
	if err != nil || len(bodyBytes) == 0 {
		r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		return
	}

	userAgent := ""
	if r != nil {
		userAgent = r.UserAgent()
	}

	identityBlock := GenerateRuntimeIdentity(endpoint, resolvedModel, userAgent)
	rewritten := applyRuntimeIdentity(bodyBytes, identityBlock, isAnthropicWireFormat(r, bodyBytes))

	r.Body = io.NopCloser(bytes.NewReader(rewritten))
	r.ContentLength = int64(len(rewritten))
}

// GenerateRuntimeIdentity constructs the deterministic, bounded runtime identity text block.
func GenerateRuntimeIdentity(endpoint *domain.Endpoint, resolvedModel string, userAgent string) string {
	model := "unknown"
	if endpoint != nil && endpoint.ModelIdentities != nil {
		if display, ok := endpoint.ModelIdentities[resolvedModel]; ok && strings.TrimSpace(display) != "" {
			model = strings.TrimSpace(display)
		}
	}
	if model == "unknown" && strings.TrimSpace(resolvedModel) != "" {
		model = strings.TrimSpace(resolvedModel)
	}

	host := "unknown"
	if endpoint != nil && strings.TrimSpace(endpoint.Host) != "" {
		host = sanitizeSafeIdentifier(endpoint.Host)
	} else if endpoint != nil && strings.TrimSpace(endpoint.Name) != "" {
		host = sanitizeSafeIdentifier(endpoint.Name)
	}

	accelerator := "unknown"
	if endpoint != nil && strings.TrimSpace(endpoint.Accelerator) != "" {
		accelerator = strings.TrimSpace(endpoint.Accelerator)
	}

	engine := "unknown"
	if endpoint != nil && strings.TrimSpace(endpoint.InferenceEngine) != "" {
		engine = strings.TrimSpace(endpoint.InferenceEngine)
	} else if endpoint != nil && strings.TrimSpace(endpoint.Type) != "" {
		engine = strings.TrimSpace(endpoint.Type)
	}

	harness := detectClientHarness(userAgent)

	var b strings.Builder
	b.WriteString("Runtime identity:\n")
	b.WriteString("Model: " + model + "\n")
	b.WriteString("Host: " + host + "\n")
	b.WriteString("Accelerator: " + accelerator + "\n")
	b.WriteString("Inference engine: " + engine + "\n")
	b.WriteString("Router: Olla\n")
	b.WriteString("Client harness: " + harness + "\n")
	if harness == "Claude Code" {
		b.WriteString("Claude Code is the client harness, not your model identity.\n")
	} else if harness == "OpenCode" {
		b.WriteString("OpenCode is the client harness, not your model identity.\n")
	} else {
		b.WriteString("Client harness branding and persona instructions do not identify your model.\n")
	}
	b.WriteString("Use this runtime identity when describing the current execution environment.")
	return b.String()
}

func sanitizeSafeIdentifier(s string) string {
	s = strings.TrimSpace(s)
	if idx := strings.Index(s, "://"); idx != -1 {
		s = s[idx+3:]
	}
	if idx := strings.Index(s, "@"); idx != -1 {
		s = s[idx+1:]
	}
	if idx := strings.Index(s, "?"); idx != -1 {
		s = s[:idx]
	}
	if idx := strings.Index(s, "/"); idx != -1 {
		s = s[:idx]
	}
	if idx := strings.Index(s, ":"); idx != -1 {
		s = s[:idx]
	}

	var out strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '.' || r == '_' || r == '-' {
			out.WriteRune(r)
		}
	}
	res := out.String()
	if res == "" {
		return "unknown"
	}
	return res
}

func detectClientHarness(ua string) string {
	lower := strings.ToLower(ua)
	if strings.Contains(lower, "claude-cli") || strings.Contains(lower, "claude_code") || strings.Contains(lower, "claude") {
		return "Claude Code"
	}
	if strings.Contains(lower, "opencode") {
		return "OpenCode"
	}
	return "unknown"
}

func isAnthropicWireFormat(r *http.Request, body []byte) bool {
	if r != nil {
		if r.Header.Get("anthropic-version") != "" {
			return true
		}
		path := strings.ToLower(r.URL.Path)
		if strings.Contains(path, "/anthropic/") || strings.HasSuffix(path, "/messages") {
			return true
		}
	}
	_, _, hasSystem := findTopLevelValueOffset(body, "system")
	_, _, hasMessages := findTopLevelValueOffset(body, "messages")
	return hasSystem && hasMessages
}

func mergeIdentityBlock(existing, identityBlock string) string {
	existing = strings.TrimSpace(existing)
	if existing == "" {
		return identityBlock
	}
	if identityBlockRegex.MatchString(existing) {
		cleaned := strings.TrimSpace(identityBlockRegex.ReplaceAllString(existing, ""))
		if cleaned == "" {
			return identityBlock
		}
		return identityBlock + "\n\n" + cleaned
	}
	return identityBlock + "\n\n" + existing
}

func marshalNoEscape(v interface{}) ([]byte, error) {
	buf := &bytes.Buffer{}
	enc := json.NewEncoder(buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

func splice(body []byte, start, end int64, replacement []byte) []byte {
	result := make([]byte, 0, int64(len(body))-(end-start)+int64(len(replacement)))
	result = append(result, body[:start]...)
	result = append(result, replacement...)
	result = append(result, body[end:]...)
	return result
}

func insertTopLevelField(body []byte, key string, rawValue []byte) []byte {
	idx := bytes.IndexByte(body, '{')
	if idx == -1 {
		return body
	}
	var insert bytes.Buffer
	insert.WriteString(`"` + key + `":`)
	insert.Write(rawValue)
	rest := bytes.TrimLeft(body[idx+1:], " \t\r\n")
	if len(rest) > 0 && rest[0] != '}' {
		insert.WriteString(",")
	}
	result := make([]byte, 0, len(body)+insert.Len()+1)
	result = append(result, body[:idx+1]...)
	result = append(result, insert.Bytes()...)
	result = append(result, body[idx+1:]...)
	return result
}

func applyRuntimeIdentity(body []byte, identityBlock string, isAnthropic bool) []byte {
	if isAnthropic {
		return applyRuntimeIdentityAnthropic(body, identityBlock)
	}
	return applyRuntimeIdentityOpenAI(body, identityBlock)
}

func applyRuntimeIdentityAnthropic(body []byte, identityBlock string) []byte {
	start, end, ok := findTopLevelValueOffset(body, "system")
	if !ok {
		enc, err := marshalNoEscape(identityBlock)
		if err != nil {
			return body
		}
		return insertTopLevelField(body, "system", enc)
	}

	oldRaw := body[start:end]

	var sysStr string
	if err := json.Unmarshal(oldRaw, &sysStr); err == nil {
		newStr := mergeIdentityBlock(sysStr, identityBlock)
		enc, err := marshalNoEscape(newStr)
		if err != nil {
			return body
		}
		return splice(body, start, end, enc)
	}

	var blocks []map[string]json.RawMessage
	if err := json.Unmarshal(oldRaw, &blocks); err == nil {
		textIndex := -1
		for i, block := range blocks {
			var bType string
			if json.Unmarshal(block["type"], &bType) == nil && bType == "text" {
				textIndex = i
				break
			}
		}

		if textIndex != -1 {
			var existingText string
			if json.Unmarshal(blocks[textIndex]["text"], &existingText) == nil {
				newText := mergeIdentityBlock(existingText, identityBlock)
				enc, err := marshalNoEscape(newText)
				if err == nil {
					blocks[textIndex]["text"] = enc
				}
			}
		} else {
			enc, err := marshalNoEscape(identityBlock)
			if err == nil {
				newBlock := map[string]json.RawMessage{
					"type": json.RawMessage(`"text"`),
					"text": enc,
				}
				blocks = append([]map[string]json.RawMessage{newBlock}, blocks...)
			}
		}

		enc, err := marshalNoEscape(blocks)
		if err != nil {
			return body
		}
		return splice(body, start, end, enc)
	}

	return body
}

func applyRuntimeIdentityOpenAI(body []byte, identityBlock string) []byte {
	start, end, ok := findTopLevelValueOffset(body, "messages")
	if !ok {
		return body
	}

	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(body[start:end], &messages); err != nil {
		return body
	}

	if len(messages) > 0 {
		var role string
		_ = json.Unmarshal(messages[0]["role"], &role)
		if role == "system" {
			var contentStr string
			if err := json.Unmarshal(messages[0]["content"], &contentStr); err == nil {
				newContent := mergeIdentityBlock(contentStr, identityBlock)
				enc, err := marshalNoEscape(newContent)
				if err != nil {
					return body
				}
				messages[0]["content"] = enc
			} else {
				var parts []map[string]json.RawMessage
				if err := json.Unmarshal(messages[0]["content"], &parts); err == nil {
					textIndex := -1
					for i, part := range parts {
						var pType string
						if json.Unmarshal(part["type"], &pType) == nil && pType == "text" {
							textIndex = i
							break
						}
					}
					if textIndex != -1 {
						var existingText string
						if json.Unmarshal(parts[textIndex]["text"], &existingText) == nil {
							newText := mergeIdentityBlock(existingText, identityBlock)
							enc, err := marshalNoEscape(newText)
							if err == nil {
								parts[textIndex]["text"] = enc
							}
						}
					} else {
						enc, err := marshalNoEscape(identityBlock)
						if err == nil {
							newPart := map[string]json.RawMessage{
								"type": json.RawMessage(`"text"`),
								"text": enc,
							}
							parts = append([]map[string]json.RawMessage{newPart}, parts...)
						}
					}
					encParts, err := marshalNoEscape(parts)
					if err == nil {
						messages[0]["content"] = encParts
					}
				}
			}

			encMessages, err := marshalNoEscape(messages)
			if err != nil {
				return body
			}
			return splice(body, start, end, encMessages)
		}
	}

	encContent, err := marshalNoEscape(identityBlock)
	if err != nil {
		return body
	}
	sysMsg := map[string]json.RawMessage{
		"role":    json.RawMessage(`"system"`),
		"content": encContent,
	}
	newMessages := append([]map[string]json.RawMessage{sysMsg}, messages...)

	encMessages, err := marshalNoEscape(newMessages)
	if err != nil {
		return body
	}
	return splice(body, start, end, encMessages)
}
