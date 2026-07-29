package cache

import (
	"math/rand"
	"testing"
)

func generateRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()
	}
	return vec
}

// BenchmarkFloat32ToBytes measures zero-copy float32 vector serialization speed.
func BenchmarkFloat32ToBytes(b *testing.B) {
	vec := generateRandomVector(VectorDim)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = Float32ToBytes(vec)
	}
}

// BenchmarkSanitizeTag measures RediSearch tag escaping speed.
func BenchmarkSanitizeTag(b *testing.B) {
	tag := "doc_wgan-123.v1@test:chunk/0"
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = SanitizeTag(tag)
	}
}
