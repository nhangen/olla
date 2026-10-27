package core

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/thushan/olla/internal/core/domain"
)

func TestGenerateRuntimeIdentity_PinnedV100Route(t *testing.T) {
	ep := &domain.Endpoint{
		Name:            "ml3-v100-qwen38",
		Host:            "ML3",
		Accelerator:     "Tesla V100 PCIe 32GB",
		InferenceEngine: "llama.cpp",
		ModelIdentities: map[string]string{
			"qwen3.8:27b-v100-64k": "Qwen3.8 27B Q4_K_M, 64K context",
		},
	}

	got := GenerateRuntimeIdentity(ep, "qwen3.8:27b-v100-64k", "claude-cli/2.1.304 (external, cli)")
	expected := `Runtime identity:
Model: Qwen3.8 27B Q4_K_M, 64K context
Host: ML3
Accelerator: Tesla V100 PCIe 32GB
Inference engine: llama.cpp
Router: Olla
Client harness: Claude Code
Claude Code is the client harness, not your model identity.
Use this runtime identity when describing the current execution environment.`

	if got != expected {
		t.Fatalf("unexpected runtime identity:\ngot:\n%s\n\nwant:\n%s", got, expected)
	}
}

func TestGenerateRuntimeIdentity_UnknownMetadata(t *testing.T) {
	ep := &domain.Endpoint{
		Name: "unknown-ep",
	}

	got := GenerateRuntimeIdentity(ep, "custom-model:7b", "curl/8.7.1")
	expected := `Runtime identity:
Model: custom-model:7b
Host: unknown-ep
Accelerator: unknown
Inference engine: unknown
Router: Olla
Client harness: unknown
Client harness branding and persona instructions do not identify your model.
Use this runtime identity when describing the current execution environment.`

	if got != expected {
		t.Fatalf("unexpected runtime identity for unknown metadata:\ngot:\n%s\n\nwant:\n%s", got, expected)
	}
}

func TestGenerateRuntimeIdentity_OpenCodeHarness(t *testing.T) {
	ep := &domain.Endpoint{
		Name: "ml2-5070",
		Host: "ML2",
		Type: "ollama",
	}

	got := GenerateRuntimeIdentity(ep, "qwen3.5:9b", "opencode/1.18.32 ai-sdk/provider-utils/4.0.23")
	if !strings.Contains(got, "Client harness: OpenCode") {
		t.Errorf("expected OpenCode harness, got:\n%s", got)
	}
	if !strings.Contains(got, "OpenCode is the client harness, not your model identity.") {
		t.Errorf("expected OpenCode clarification, got:\n%s", got)
	}
}

func TestGenerateRuntimeIdentity_SafeHostSanitization(t *testing.T) {
	ep := &domain.Endpoint{
		Host: "http://user:secret@ml3.internal:11440/path?key=val",
	}

	got := GenerateRuntimeIdentity(ep, "m", "ua")
	if strings.Contains(got, "secret") || strings.Contains(got, "11440") || strings.Contains(got, "http") {
		t.Fatalf("sensitive info leaked into identity block:\n%s", got)
	}
	if !strings.Contains(got, "Host: ml3.internal") {
		t.Errorf("expected sanitized host ml3.internal, got:\n%s", got)
	}
}

func TestInjectRuntimeIdentity_OpenAI_StringSystemMessage(t *testing.T) {
	ep := &domain.Endpoint{
		Host:            "ML3",
		Accelerator:     "Tesla V100 PCIe 32GB",
		InferenceEngine: "llama.cpp",
	}

	input := `{"model":"local-long-v100","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Hello"}]}`
	req, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/chat/completions", io.NopCloser(strings.NewReader(input)))
	req.Header.Set("User-Agent", "claude-cli/2.1.304")
	req.ContentLength = int64(len(input))

	InjectRuntimeIdentity(context.Background(), req, ep, "qwen3.8:27b-v100-64k")

	body, _ := io.ReadAll(req.Body)
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("invalid json: %v", err)
	}

	messages := parsed["messages"].([]interface{})
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}
	sysMsg := messages[0].(map[string]interface{})
	if sysMsg["role"] != "system" {
		t.Fatalf("expected system message at index 0, got %v", sysMsg["role"])
	}
	content := sysMsg["content"].(string)
	if !strings.HasPrefix(content, "Runtime identity:\nModel: qwen3.8:27b-v100-64k") {
		t.Fatalf("content missing runtime identity prefix: %s", content)
	}
	if !strings.HasSuffix(content, "You are a helpful assistant.") {
		t.Fatalf("original system content lost: %s", content)
	}
}

func TestInjectRuntimeIdentity_OpenAI_AbsentSystemMessage(t *testing.T) {
	ep := &domain.Endpoint{
		Host: "ML3",
	}

	input := `{"model":"local-long-v100","messages":[{"role":"user","content":"Hello"}]}`
	req, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/chat/completions", io.NopCloser(strings.NewReader(input)))
	req.ContentLength = int64(len(input))

	InjectRuntimeIdentity(context.Background(), req, ep, "qwen3.8:27b-v100-64k")

	body, _ := io.ReadAll(req.Body)
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("invalid json: %v", err)
	}

	messages := parsed["messages"].([]interface{})
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages (system + user), got %d", len(messages))
	}
	sysMsg := messages[0].(map[string]interface{})
	if sysMsg["role"] != "system" {
		t.Fatalf("expected system message at index 0, got %v", sysMsg["role"])
	}
	userMsg := messages[1].(map[string]interface{})
	if userMsg["role"] != "user" || userMsg["content"] != "Hello" {
		t.Fatalf("user message corrupted: %v", userMsg)
	}
}

func TestInjectRuntimeIdentity_OpenAI_PreservesToolContinuations(t *testing.T) {
	ep := &domain.Endpoint{
		Host: "ML3",
	}

	input := `{"model":"m","messages":[{"role":"system","content":"sys"},{"role":"user","content":"call tool"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Read","arguments":"{\"path\":\"a.txt\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"file contents"}]}`
	req, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/chat/completions", io.NopCloser(strings.NewReader(input)))
	req.ContentLength = int64(len(input))

	InjectRuntimeIdentity(context.Background(), req, ep, "resolved-model")

	body, _ := io.ReadAll(req.Body)
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("invalid json: %v", err)
	}

	messages := parsed["messages"].([]interface{})
	if len(messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(messages))
	}
	// Tool sequence must be strictly assistant at 2, tool at 3
	asst := messages[2].(map[string]interface{})
	if asst["role"] != "assistant" || asst["tool_calls"] == nil {
		t.Fatalf("assistant tool_calls corrupted: %v", asst)
	}
	tool := messages[3].(map[string]interface{})
	if tool["role"] != "tool" || tool["tool_call_id"] != "call_1" {
		t.Fatalf("tool continuation corrupted: %v", tool)
	}
}

func TestInjectRuntimeIdentity_Anthropic_TopLevelSystem(t *testing.T) {
	ep := &domain.Endpoint{
		Host: "ML3",
	}

	input := `{"model":"m","system":"You are Claude.","messages":[{"role":"user","content":"Hello"}]}`
	req, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/messages", io.NopCloser(strings.NewReader(input)))
	req.Header.Set("anthropic-version", "2023-06-01")
	req.ContentLength = int64(len(input))

	InjectRuntimeIdentity(context.Background(), req, ep, "resolved-model")

	body, _ := io.ReadAll(req.Body)
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("invalid json: %v", err)
	}

	sys := parsed["system"].(string)
	if !strings.HasPrefix(sys, "Runtime identity:\nModel: resolved-model") {
		t.Fatalf("system missing identity prefix: %s", sys)
	}
	if !strings.HasSuffix(sys, "You are Claude.") {
		t.Fatalf("system lost original prompt: %s", sys)
	}

	// Verify messages array was NOT modified (no system message added)
	messages := parsed["messages"].([]interface{})
	if len(messages) != 1 {
		t.Fatalf("messages altered in Anthropic request: %d", len(messages))
	}
	if messages[0].(map[string]interface{})["role"] != "user" {
		t.Fatalf("messages[0] must remain user turn in Anthropic request")
	}
}

func TestInjectRuntimeIdentity_Anthropic_AbsentSystem(t *testing.T) {
	ep := &domain.Endpoint{
		Host: "ML3",
	}

	input := `{"model":"m","messages":[{"role":"user","content":"Hello"}]}`
	req, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/messages", io.NopCloser(strings.NewReader(input)))
	req.Header.Set("anthropic-version", "2023-06-01")
	req.ContentLength = int64(len(input))

	InjectRuntimeIdentity(context.Background(), req, ep, "resolved-model")

	body, _ := io.ReadAll(req.Body)
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("invalid json: %v", err)
	}

	if parsed["system"] == nil {
		t.Fatalf("expected top-level system field in Anthropic request")
	}
	sys := parsed["system"].(string)
	if !strings.HasPrefix(sys, "Runtime identity:\nModel: resolved-model") {
		t.Fatalf("unexpected system content: %s", sys)
	}
}

func TestInjectRuntimeIdentity_Anthropic_ContentBlocksWithCacheControl(t *testing.T) {
	ep := &domain.Endpoint{
		Host: "ML3",
	}

	input := `{"model":"m","system":[{"type":"text","text":"Base instruction","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"Hello"}]}`
	req, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/messages", io.NopCloser(strings.NewReader(input)))
	req.Header.Set("anthropic-version", "2023-06-01")
	req.ContentLength = int64(len(input))

	InjectRuntimeIdentity(context.Background(), req, ep, "resolved-model")

	body, _ := io.ReadAll(req.Body)
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("invalid json: %v", err)
	}

	blocks := parsed["system"].([]interface{})
	if len(blocks) != 1 {
		t.Fatalf("expected 1 system block, got %d", len(blocks))
	}
	block := blocks[0].(map[string]interface{})
	text := block["text"].(string)
	if !strings.HasPrefix(text, "Runtime identity:") {
		t.Fatalf("identity not merged into block text: %s", text)
	}
	if !strings.HasSuffix(text, "Base instruction") {
		t.Fatalf("base instruction lost: %s", text)
	}
	// cache_control must be preserved untouched!
	if block["cache_control"] == nil {
		t.Fatalf("cache_control was stripped or lost")
	}
}

func TestInjectRuntimeIdentity_ReplacesStaleIdentity(t *testing.T) {
	ep1 := &domain.Endpoint{
		Host: "ML1",
	}
	ep2 := &domain.Endpoint{
		Host: "ML3",
	}

	// Turn 1 on ep1
	input := `{"model":"m","messages":[{"role":"system","content":"You are a bot."},{"role":"user","content":"hi"}]}`
	req1, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/chat/completions", io.NopCloser(strings.NewReader(input)))
	req1.ContentLength = int64(len(input))
	InjectRuntimeIdentity(context.Background(), req1, ep1, "model-1")

	body1, _ := io.ReadAll(req1.Body)

	// Turn 2 or failover to ep2
	req2, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/chat/completions", io.NopCloser(bytes.NewReader(body1)))
	req2.ContentLength = int64(len(body1))
	InjectRuntimeIdentity(context.Background(), req2, ep2, "model-2")

	body2, _ := io.ReadAll(req2.Body)

	var parsed map[string]interface{}
	_ = json.Unmarshal(body2, &parsed)
	messages := parsed["messages"].([]interface{})
	sysContent := messages[0].(map[string]interface{})["content"].(string)

	count := strings.Count(sysContent, "Runtime identity:")
	if count != 1 {
		t.Fatalf("identity block accumulated on retry/failover: count=%d\ncontent:\n%s", count, sysContent)
	}
	if !strings.Contains(sysContent, "Host: ML3") {
		t.Fatalf("new host ML3 missing from replaced identity block:\n%s", sysContent)
	}
	if strings.Contains(sysContent, "Host: ML1") {
		t.Fatalf("stale host ML1 still present after replacement:\n%s", sysContent)
	}
}

func TestInjectRuntimeIdentity_PreservesHTMLTagsWithoutEscaping(t *testing.T) {
	ep := &domain.Endpoint{
		Host: "ML3",
	}

	input := `{"model":"m","messages":[{"role":"system","content":"<task id=\"1\">Check a < b & c > d</task>"}]}`
	req, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/chat/completions", io.NopCloser(strings.NewReader(input)))
	req.ContentLength = int64(len(input))

	InjectRuntimeIdentity(context.Background(), req, ep, "resolved-model")

	body, _ := io.ReadAll(req.Body)
	if bytes.Contains(body, []byte(`\u003c`)) || bytes.Contains(body, []byte(`\u003e`)) || bytes.Contains(body, []byte(`\u0026`)) {
		t.Fatalf("HTML tags were escaped to unicode:\n%s", string(body))
	}
	if !bytes.Contains(body, []byte(`<task id=\"1\">Check a < b & c > d</task>`)) {
		t.Fatalf("raw XML content was modified:\n%s", string(body))
	}
}

// --- Issue #1068: gaps the original test file did not cover ---
//
// These tests drive the real injection path (InjectRuntimeIdentity) for the
// contract areas flagged in https://github.com/nhangen/llm-tools/issues/1068:
//   - OpenAI messages[0] content-parts list
//   - stale-block replacement when user text merely QUOTES "Runtime identity:"
//   - Content-Length resync on retry body reset
//   - byte preservation of <, >, & and unicode in BOTH user text AND the
//     surrounding JSON bytes (byte-splice, not full re-marshal)
//   - documented whitespace handling of the user's system text

func TestInjectRuntimeIdentity_OpenAI_ContentPartsList(t *testing.T) {
	ep := &domain.Endpoint{Host: "ML3"}

	// messages[0] is a system message whose content is a list of content parts,
	// with a non-text part (image_url) that must survive untouched.
	input := `{"model":"m","messages":[{"role":"system","content":[{"type":"text","text":"You are a helpful assistant."},{"type":"image_url","image_url":{"url":"http://example.com/a.png"}}]},{"role":"user","content":"describe it"}]}`
	req, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/chat/completions", io.NopCloser(strings.NewReader(input)))
	req.ContentLength = int64(len(input))

	InjectRuntimeIdentity(context.Background(), req, ep, "qwen3.8:27b-v100-64k")

	body, _ := io.ReadAll(req.Body)
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("invalid json: %v\n%s", err, body)
	}

	messages := parsed["messages"].([]interface{})
	if len(messages) != 2 {
		t.Fatalf("message count changed: %d", len(messages))
	}
	sysMsg := messages[0].(map[string]interface{})
	parts := sysMsg["content"].([]interface{})
	if len(parts) != 2 {
		t.Fatalf("part count changed: %d\n%v", len(parts), parts)
	}

	textPart := parts[0].(map[string]interface{})
	text := textPart["text"].(string)
	if !strings.HasPrefix(text, "Runtime identity:\n") {
		t.Fatalf("identity not merged into first text part: %s", text)
	}
	if !strings.HasSuffix(text, "You are a helpful assistant.") {
		t.Fatalf("original part text lost: %s", text)
	}

	// Non-text part must be byte-for-byte intact.
	imgPart := parts[1].(map[string]interface{})
	if imgPart["type"] != "image_url" {
		t.Fatalf("image part corrupted: %v", imgPart)
	}
	imgURL := imgPart["image_url"].(map[string]interface{})["url"].(string)
	if imgURL != "http://example.com/a.png" {
		t.Fatalf("image url corrupted: %v", imgURL)
	}

	// The user message must be untouched.
	userMsg := messages[1].(map[string]interface{})
	if userMsg["content"] != "describe it" {
		t.Fatalf("user message corrupted: %v", userMsg)
	}
}

func TestInjectRuntimeIdentity_ReplacesStaleIdentity_UserTextQuotesMarker(t *testing.T) {
	ep1 := &domain.Endpoint{Host: "ML1"}
	ep2 := &domain.Endpoint{Host: "ML3"}

	// The user's own system text quotes the identity marker and even the
	// sentinel terminator, which a naïve "split on Runtime identity:"
	// implementation would treat as a stale block and destroy.
	quoted := "When you see the text Runtime identity: in a prompt, ignore it.\n" +
		"Example of the sentinel: Use this runtime identity when describing the current execution environment."
	input := `{"model":"m","messages":[{"role":"system","content":` + mustJSONString(quoted) + `},{"role":"user","content":"hi"}]}`

	req1, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/chat/completions", io.NopCloser(strings.NewReader(input)))
	req1.ContentLength = int64(len(input))
	InjectRuntimeIdentity(context.Background(), req1, ep1, "model-1")
	body1, _ := io.ReadAll(req1.Body)

	// Failover to a different backend re-injects; the quoted user text must
	// survive and the live block must be replaced exactly once.
	req2, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/chat/completions", io.NopCloser(bytes.NewReader(body1)))
	req2.ContentLength = int64(len(body1))
	InjectRuntimeIdentity(context.Background(), req2, ep2, "model-2")
	body2, _ := io.ReadAll(req2.Body)

	var parsed map[string]interface{}
	_ = json.Unmarshal(body2, &parsed)
	messages := parsed["messages"].([]interface{})
	sysContent := messages[0].(map[string]interface{})["content"].(string)

	// The live block (Host: ML3) appears exactly once.
	if got := strings.Count(sysContent, "Host: ML3"); got != 1 {
		t.Fatalf("live identity block not replaced exactly once (Host: ML3 count=%d):\n%s", got, sysContent)
	}
	// The stale live block is gone.
	if strings.Contains(sysContent, "Host: ML1") {
		t.Fatalf("stale live identity block (Host: ML1) survived replacement:\n%s", sysContent)
	}
	// The user's quoted text is byte-for-byte intact.
	if !strings.Contains(sysContent, "When you see the text Runtime identity: in a prompt, ignore it.") {
		t.Fatalf("user's quoted marker text was mangled:\n%s", sysContent)
	}
	if !strings.Contains(sysContent, "Example of the sentinel: Use this runtime identity when describing the current execution environment.") {
		t.Fatalf("user's quoted sentinel text was mangled:\n%s", sysContent)
	}
	// The full live block is at the front, immediately followed by the
	// (intact) user text — no accumulation, no reordering.
	if !strings.HasPrefix(sysContent, "Runtime identity:\nModel: model-2\nHost: ML3") {
		t.Fatalf("live block not leading:\n%s", sysContent)
	}
}

func TestInjectRuntimeIdentity_ContentLengthResyncOnRetryBodyReset(t *testing.T) {
	ep1 := &domain.Endpoint{Host: "ML1"}
	ep2 := &domain.Endpoint{Host: "ML3"}

	original := `{"model":"m","messages":[{"role":"system","content":"You are a bot."},{"role":"user","content":"hi"}]}`

	// Attempt 1: the injector consumes the body and re-emits a longer one.
	req, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/chat/completions", io.NopCloser(strings.NewReader(original)))
	req.ContentLength = int64(len(original))
	InjectRuntimeIdentity(context.Background(), req, ep1, "model-1")

	// Retry body reset (mirrors RetryHandler.preserveRequestBody): the proxy
	// drains the body to preserve the bytes, then re-wraps them. After the
	// injector has grown the body, the preserved bytes ARE the rewritten body,
	// and ContentLength must track them.
	bodyBytes := mustReadAll(t, req)
	req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
	req.ContentLength = int64(len(bodyBytes))

	// Attempt 2 (failover) re-injects against the already-rewritten body.
	InjectRuntimeIdentity(context.Background(), req, ep2, "model-2")

	final, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("reading final body: %v", err)
	}
	if req.ContentLength != int64(len(final)) {
		t.Fatalf("ContentLength (%d) != actual body length (%d) after retry body reset; the backend would read a truncated/garbled request",
			req.ContentLength, len(final))
	}

	var parsed map[string]interface{}
	_ = json.Unmarshal(final, &parsed)
	sysContent := parsed["messages"].([]interface{})[0].(map[string]interface{})["content"].(string)
	if strings.Count(sysContent, "Runtime identity:") != 1 {
		t.Fatalf("identity accumulated across retries: %s", sysContent)
	}
}

func TestInjectRuntimeIdentity_PreservesUnicodeAndSpecialBytes(t *testing.T) {
	ep := &domain.Endpoint{Host: "ML3"}

	// User text carrying <, >, &, and non-ASCII (including a supplementary-plane
	// rune) that must survive byte-for-byte in BOTH the JSON payload and the
	// surrounding bytes (byte-splice, not re-marshal of the whole body).
	special := "Café — naïve: a < b && c > d — 日本語テスト — 👍"
	input := `{"model":"m","messages":[{"role":"system","content":` + mustJSONString(special) + `},{"role":"user","content":"compare <x> & <y>"}]}`
	req, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/chat/completions", io.NopCloser(strings.NewReader(input)))
	req.ContentLength = int64(len(input))

	InjectRuntimeIdentity(context.Background(), req, ep, "qwen3.8:27b-v100-64k")

	body, _ := io.ReadAll(req.Body)

	// No HTML-escaping of the specials anywhere in the payload.
	for _, escaped := range []string{`&#x3c;`, `&#x3e;`, `&#x26;`} {
		if bytes.Contains(body, []byte(escaped)) {
			t.Fatalf("special characters were HTML/unicode-escaped in payload:\n%s", body)
		}
	}
	// IMPORTANT (documented contract): the OpenAI path splices at the TOP-LEVEL
	// "messages" value, so the whole messages array is re-marshalled
	// (applyRuntimeIdentityOpenAI -> marshalNoEscape). Untouched fields like
	// "model" keep their exact input bytes; untouched message objects do NOT —
	// they are re-encoded with Go's json key ordering (content before role)
	// while preserving values byte-for-byte after JSON decoding. The assertion
	// below checks value-level preservation, which is what the wire contract
	// actually guarantees; a byte-equality check would over-specify.
	var parsed map[string]interface{}
	_ = json.Unmarshal(body, &parsed)
	msgs := parsed["messages"].([]interface{})
	sysContent := msgs[0].(map[string]interface{})["content"].(string)
	if !strings.Contains(sysContent, special) {
		t.Fatalf("system content lost special text:\n%s", sysContent)
	}
	if msgs[1].(map[string]interface{})["content"] != "compare <x> & <y>" {
		t.Fatalf("user message corrupted: %v", msgs[1])
	}

	// Untouched top-level field keeps its exact original bytes (true byte-splice
	// invariant): the "model" field is not touched by the OpenAI path.
	if !bytes.Contains(body, []byte(`"model":"m"`)) {
		t.Fatalf("untouched top-level field bytes were rewritten:\n%s", body)
	}
	// The identity block is byte-exact in the payload (JSON-encoded, with its
	// opening quote; the closing quote comes after the user text so we match
	// the prefix that is guaranteed contiguous).
	identity := GenerateRuntimeIdentity(ep, "qwen3.8:27b-v100-64k", "")
	identityJSON := mustJSONString(identity)
	if !bytes.Contains(body, []byte(identityJSON[:len(identityJSON)-1])) {
		t.Fatalf("identity block not byte-exact in payload:\n%s", body)
	}
}

// TestInjectRuntimeIdentity_SystemWhitespaceContract documents the CURRENT
// contract for user system-text whitespace (#1068 asks for a decision here).
//
// Go mergeIdentityBlock runs strings.TrimSpace on the user text, so leading and
// trailing whitespace in the user's system prompt is STRIPPED. This matches the
// Python spec-mirror merge_identity in llm-tools tests/ and is idempotency-safe
// (the "\n\n" separator is unambiguous regardless of the user's original
// padding). The trim is per-request and deterministic, so the backend-visible
// prefix is byte-stable across turns and prompt-cache reuse is unaffected by
// the trim; the once-only prefix invalidation comes from the injected block
// itself, not from the trim.
func TestInjectRuntimeIdentity_SystemWhitespaceContract(t *testing.T) {
	ep := &domain.Endpoint{Host: "ML3"}

	padded := "  \n  You are a bot. \n  "
	input := `{"model":"m","messages":[{"role":"system","content":` + mustJSONString(padded) + `},{"role":"user","content":"hi"}]}`
	req, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/chat/completions", io.NopCloser(strings.NewReader(input)))
	req.ContentLength = int64(len(input))

	InjectRuntimeIdentity(context.Background(), req, ep, "m1")
	body, _ := io.ReadAll(req.Body)

	var parsed map[string]interface{}
	_ = json.Unmarshal(body, &parsed)
	content := parsed["messages"].([]interface{})[0].(map[string]interface{})["content"].(string)

	if strings.Contains(content, "\n\n\n") {
		t.Fatalf("separator whitespace is not stable:\n%q", content)
	}
	if !strings.HasSuffix(content, "You are a bot.") {
		t.Fatalf("user text lost after trim:\n%q", content)
	}
	if !strings.Contains(content, "execution environment.\n\nYou are a bot.") {
		t.Fatalf("expected identity block, blank line, then trimmed user text:\n%q", content)
	}

	// Idempotency: re-injecting must not grow or shift the user text.
	req2, _ := http.NewRequestWithContext(context.Background(), "POST", "/v1/chat/completions", io.NopCloser(bytes.NewReader(body)))
	req2.ContentLength = int64(len(body))
	InjectRuntimeIdentity(context.Background(), req2, ep, "m1")
	body2, _ := io.ReadAll(req2.Body)
	var parsed2 map[string]interface{}
	_ = json.Unmarshal(body2, &parsed2)
	content2 := parsed2["messages"].([]interface{})[0].(map[string]interface{})["content"].(string)
	if content2 != content {
		t.Fatalf("re-injection changed the merged system text (trim is not idempotent):\nfirst:\n%q\nsecond:\n%q", content, content2)
	}
}

// mustJSONString encodes s as a JSON string with HTML escaping disabled,
// matching the encoder (marshalNoEscape) the injector uses. Note: plain
// json.Marshal HTML-escapes <, >, & — the byte-preservation assertions depend
// on the unescaped form.
func mustJSONString(s string) string {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(s); err != nil {
		panic(err)
	}
	return strings.TrimSuffix(buf.String(), "\n")
}

func mustReadAll(t *testing.T, req *http.Request) []byte {
	t.Helper()
	b, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("reading body: %v", err)
	}
	return b
}
