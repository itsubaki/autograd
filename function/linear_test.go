package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleLinear() {
	x := variable.NewOf(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)
	w := variable.NewOf(
		[]float64{1, 2, 3, 4},
		[]float64{5, 6, 7, 8},
		[]float64{9, 10, 11, 12},
	)

	y := F.Linear(x, w)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)
	fmt.Println(w.Grad)

	// Output:
	// variable([[38 44 50 56] [83 98 113 128]])
	// variable([[10 26 42] [10 26 42]])
	// variable([[5 5 5 5] [7 7 7 7] [9 9 9 9]])
}

func ExampleLinear_bias() {
	x := variable.NewOf(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)
	w := variable.NewOf(
		[]float64{1, 2, 3, 4},
		[]float64{5, 6, 7, 8},
		[]float64{9, 10, 11, 12},
	)
	b := variable.New(1.0)

	y := F.Linear(x, w, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)
	fmt.Println(w.Grad)
	fmt.Println(b.Grad)

	// Output:
	// variable([[39 45 51 57] [84 99 114 129]])
	// variable([[10 26 42] [10 26 42]])
	// variable([[5 5 5 5] [7 7 7 7] [9 9 9 9]])
	// variable([8])
}
