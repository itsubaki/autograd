//go:build darwin

package tensor_test

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/itsubaki/autograd/tensor"
)

func ExampleMatMul2D() {
	a := []float64{
		1, 2,
		3, 4,
	}

	b := []float64{
		5, 6,
		7, 8,
	}

	o := make([]float64, 4)
	tensor.MatMul2D(a, b, o, 2, 2, 2)

	fmt.Printf("%.0f %.0f\n", o[0], o[1])
	fmt.Printf("%.0f %.0f\n", o[2], o[3])

	// Output:
	// 19 22
	// 43 50
}

func benchmarkMatMul2D(b *testing.B, m, n, k int) {
	a := make([]float64, m*n)
	c := make([]float64, n*k)
	o := make([]float64, m*k)

	for i := range a {
		a[i] = rand.Float64()
	}

	for i := range c {
		c[i] = rand.Float64()
	}

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		tensor.MatMul2D(a, c, o, m, n, k)
	}
}

func BenchmarkMatMul2D_1024(b *testing.B) {
	benchmarkMatMul2D(b, 1024, 1024, 1024)
}

func BenchmarkMatMul2D_2048(b *testing.B) {
	benchmarkMatMul2D(b, 2048, 2048, 2048)
}
