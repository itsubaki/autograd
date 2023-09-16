package vector_test

import (
	"fmt"

	"github.com/itsubaki/autograd/vector"
)

func ExampleNewLike() {
	v := []float64{1, 2, 3, 4, 5}
	fmt.Println(vector.NewLike(v))

	// Output:
	// [0 0 0 0 0]

}

func ExampleOneLike() {
	v := []float64{1, 2, 3, 4, 5}
	fmt.Println(vector.OneLike(v))

	// Output:
	// [1 1 1 1 1]
}

func ExampleF() {
	v := []float64{1, 2, 3, 4, 5}
	fmt.Println(vector.F(v, func(a float64) float64 { return a * a }))

	// Output:
	// [1 4 9 16 25]
}

func ExampleF2() {
	v := []float64{1, 2, 3, 4, 5}
	w := []float64{6, 7, 8, 9, 10}
	fmt.Println(vector.F2(v, w, func(a, b float64) float64 { return a * b }))

	// Output:
	// [6 14 24 36 50]
}
