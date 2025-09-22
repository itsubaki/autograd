package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleLinear() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	w := variable.New(
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	).Reshape(3, 4)

	y := F.Linear(x, w)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)
	fmt.Println(w.Grad)

	// Output:
	// variable[2 4]([38 44 50 56 83 98 113 128])
	// variable[2 3]([10 26 42 10 26 42])
	// variable[3 4]([5 5 5 5 7 7 7 7 9 9 9 9])
}

func ExampleLinear_bias() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	w := variable.New(
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	).Reshape(3, 4)

	b := variable.New(1.0)

	y := F.Linear(x, w, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)
	fmt.Println(w.Grad)
	fmt.Println(b.Grad)

	// Output:
	// variable[2 4]([39 45 51 57 84 99 114 129])
	// variable[2 3]([10 26 42 10 26 42])
	// variable[3 4]([5 5 5 5 7 7 7 7 9 9 9 9])
	// variable(8)
}

func ExampleLinear_batch() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,

		7, 8, 9,
		10, 11, 12,
	).Reshape(2, 2, 3)

	w := variable.New(
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	).Reshape(1, 3, 4)

	y := F.Linear(x, w)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)
	fmt.Println(w.Grad)

	// Output:
	// variable[2 2 4]([38 44 50 56 83 98 113 128 128 152 176 200 173 206 239 272])
	// variable[2 2 3]([10 26 42 10 26 42 10 26 42 10 26 42])
	// variable[1 3 4]([22 22 22 22 26 26 26 26 30 30 30 30])
}

func ExampleLinear_batchx() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(1, 2, 3)

	w := variable.New(
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,

		13, 14, 15, 16,
		17, 18, 19, 20,
		21, 22, 23, 24,
	).Reshape(2, 3, 4)

	y := F.Linear(x, w)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)
	fmt.Println(w.Grad)

	// Output:
	// variable[2 2 4]([38 44 50 56 83 98 113 128 110 116 122 128 263 278 293 308])
	// variable[1 2 3]([68 100 132 68 100 132])
	// variable[2 3 4]([5 5 5 5 7 7 7 7 9 9 9 9 5 5 5 5 7 7 7 7 9 9 9 9])
}

func Example_equal() {
	fmt.Println(F.Equal([]int{1, 2, 3}, []int{1, 2, 3}))
	fmt.Println(F.Equal([]int{1, 2, 3}, []int{3, 2, 1}))
	fmt.Println(F.Equal([]int{1, 2, 3}, []int{1, 2}))

	// Output:
	// true
	// false
	// false
}
