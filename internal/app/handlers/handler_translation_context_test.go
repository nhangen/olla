package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/thushan/olla/internal/adapter/inspector"
	"github.com/thushan/olla/internal/adapter/translator/anthropic"
	"github.com/thushan/olla/internal/config"
	"github.com/thushan/olla/internal/core/domain"
	"github.com/thushan/olla/internal/core/ports"
	"github.com/thushan/olla/internal/logger"
)

func contextRecoveryHandler(t *testing.T, backend func(http.ResponseWriter, *http.Request) error) http.HandlerFunc {
	t.Helper()
	log := &mockStyledLogger{}
	trans, err := anthropic.NewTranslator(log, config.AnthropicTranslatorConfig{MaxMessageSize: 10 << 20})
	require.NoError(t, err)
	app := &Application{
		logger: log,
		proxyService: &mockProxyService{proxyFunc: func(_ context.Context, w http.ResponseWriter, r *http.Request, _ []*domain.Endpoint, _ *ports.RequestStats, _ logger.StyledLogger) error {
			return backend(w, r)
		}},
		statsCollector:   &mockStatsCollector{},
		repository:       &mockEndpointRepository{},
		inspectorChain:   inspector.NewChain(log),
		profileFactory:   &mockProfileFactory{},
		discoveryService: &mockDiscoveryServiceForTranslation{},
		Config:           &config.Config{},
	}
	return app.translationHandler(trans)
}

func contextRecoveryRequest(stream bool) *http.Request {
	body := fmt.Sprintf(`{"model":"test-model","max_tokens":1024,"stream":%t,"messages":[{"role":"user","content":"hello"}]}`, stream)
	return httptest.NewRequest(http.MethodPost, "/olla/anthropic/v1/messages", strings.NewReader(body))
}

func TestTranslationHandler_ContextRecoveryUsage(t *testing.T) {
	var captured map[string]interface{}
	handler := contextRecoveryHandler(t, func(w http.ResponseWriter, r *http.Request) error {
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			return err
		}
		options, _ := captured["stream_options"].(map[string]interface{})
		if options["include_usage"] != true {
			w.WriteHeader(http.StatusBadRequest)
			return json.NewEncoder(w).Encode(map[string]interface{}{"error": map[string]interface{}{"message": "streaming usage was not requested"}})
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := io.WriteString(w, "data: {\"id\":\"usage-probe\",\"model\":\"test-model\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"+
			"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"+
			"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":65400,\"completion_tokens\":93,\"total_tokens\":65493}}\n\n"+
			"data: [DONE]\n\n")
		return err
	})
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, contextRecoveryRequest(true))
	require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
	assert.Equal(t, true, captured["stream"])
	assert.Contains(t, recorder.Body.String(), "event: message_stop")
	var delta map[string]interface{}
	for _, event := range strings.Split(recorder.Body.String(), "\n\n") {
		if strings.HasPrefix(event, "event: message_delta\n") {
			require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(event, "event: message_delta\ndata: ")), &delta))
		}
	}
	require.NotNil(t, delta)
	usage, ok := delta["usage"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, float64(65400), usage["input_tokens"])
	assert.Equal(t, float64(93), usage["output_tokens"])
}

func TestTranslationHandler_ContextRecoveryNonStreamingOptions(t *testing.T) {
	var captured map[string]interface{}
	handler := contextRecoveryHandler(t, func(w http.ResponseWriter, r *http.Request) error {
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			return err
		}
		_, err := io.WriteString(w, `{"id":"nonstream","model":"test-model","choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}`)
		return err
	})
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, contextRecoveryRequest(false))
	require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
	assert.NotContains(t, captured, "stream_options")
}

func TestTranslationHandler_ContextRecoveryErrors(t *testing.T) {
	overflow := "request (65729 tokens) exceeds the available context size (65536 tokens), try increasing it"
	for _, stream := range []bool{false, true} {
		for _, tc := range []struct {
			name    string
			message string
			status  int
			want    string
		}{
			{"overflow", overflow, 400, "prompt is too long: " + overflow},
			{"equal_counts", "request (65536 tokens) exceeds the available context size (65536 tokens), try increasing it", 400, ""},
			{"smaller_count", "request (10 tokens) exceeds the available context size (20 tokens), try increasing it", 400, ""},
			{"non400", overflow, 500, ""},
			{"quoted_text", "user said: " + overflow, 400, ""},
			{"other_error", "Invalid temperature value", 400, ""},
			{"numeric_overflow", "request (99999999999999999999999999 tokens) exceeds the available context size (65536 tokens), try increasing it", 400, ""},
		} {
			t.Run(fmt.Sprintf("%s/stream=%t", tc.name, stream), func(t *testing.T) {
				handler := contextRecoveryHandler(t, func(w http.ResponseWriter, _ *http.Request) error {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(tc.status)
					return json.NewEncoder(w).Encode(map[string]interface{}{"error": map[string]interface{}{"message": tc.message}})
				})
				recorder := httptest.NewRecorder()
				handler.ServeHTTP(recorder, contextRecoveryRequest(stream))
				require.Equal(t, tc.status, recorder.Code)
				var body struct {
					Type  string `json:"type"`
					Error struct {
						Type    string `json:"type"`
						Message string `json:"message"`
					} `json:"error"`
				}
				require.NoError(t, json.NewDecoder(bytes.NewReader(recorder.Body.Bytes())).Decode(&body))
				assert.Equal(t, "error", body.Type)
				if tc.status == 400 {
					assert.Equal(t, "invalid_request_error", body.Error.Type)
				} else {
					assert.Equal(t, "api_error", body.Error.Type)
				}
				want := tc.want
				if want == "" {
					want = tc.message
				}
				assert.Equal(t, want, body.Error.Message)
			})
		}
	}
}
