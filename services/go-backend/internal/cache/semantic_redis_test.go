package cache

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func TestFloat32ToBytes(t *testing.T) {
	vec := []float32{1.0, -0.5, 0.0, 3.14159}
	buf := Float32ToBytes(vec)

	if len(buf) != len(vec)*4 {
		t.Fatalf("expected buffer length %d, got %d", len(vec)*4, len(buf))
	}

	for i, orig := range vec {
		bits := binary.LittleEndian.Uint32(buf[i*4 : (i+1)*4])
		val := math.Float32frombits(bits)
		if val != orig {
			t.Errorf("at index %d: expected %f, got %f", i, orig, val)
		}
	}
}

func TestFloat32ToBytes_Empty(t *testing.T) {
	buf := Float32ToBytes(nil)
	if len(buf) != 0 {
		t.Errorf("expected 0 bytes for nil vector, got %d", len(buf))
	}
}

func TestFloat32ToBytes_Dimensions(t *testing.T) {
	vec := make([]float32, 384)
	for i := range vec {
		vec[i] = float32(i) * 0.001
	}

	buf := Float32ToBytes(vec)
	if len(buf) != 384*4 {
		t.Fatalf("expected 1536 bytes for 384-dim vector, got %d", len(buf))
	}

	var reconstructed float32
	err := binary.Read(bytes.NewReader(buf[:4]), binary.LittleEndian, &reconstructed)
	if err != nil {
		t.Fatalf("binary read failed: %v", err)
	}
	if reconstructed != vec[0] {
		t.Errorf("expected first element %f, got %f", vec[0], reconstructed)
	}
}

func TestSanitizeTag(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"doc123", "doc123"},
		{"doc-123.pdf", "doc\\-123\\.pdf"},
		{"user@domain:1", "user\\@domain\\:1"},
		{"path/to/doc", "path\\/to\\/doc"},
		{"complex-{tag}", "complex\\-\\{tag\\}"},
	}

	for _, tt := range tests {
		result := SanitizeTag(tt.input)
		if result != tt.expected {
			t.Errorf("SanitizeTag(%q) = %q; expected %q", tt.input, result, tt.expected)
		}
	}
}
