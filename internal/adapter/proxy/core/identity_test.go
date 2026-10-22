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
