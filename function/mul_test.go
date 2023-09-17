package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func ExampleMul() {
	// p139
	a := variable.New(3.0)
	b := variable.New(2.0)
	c := variable.New(1.0)

	y := F.Add(F.Mul(a, b), c)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad)
	fmt.Println(b.Grad)

	// Output:
	// variable([7])
	// [2]
	// [3]
}

func ExampleMulT() {
	v := variable.New(3.0)
	w := variable.New(2.0)
	fmt.Println(v)
	fmt.Println(w)

	f := F.MulT{}
	fmt.Println(f.Forward(v.Data, w.Data))
	fmt.Println(f.Backward(vector.OneLike(v.Data), vector.OneLike(w.Data)))

	// Output:
	// variable([3])
	// variable([2])
	// [[6]]
	// [[2] [3]]
}
