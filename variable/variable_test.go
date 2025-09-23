package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/rand"
	"github.com/itsubaki/autograd/variable"
)

func ExampleVariable() {
	v := variable.New(1, 2, 3, 4)
	fmt.Println(v)

	// Output:
	// variable[4]([1 2 3 4])
}

func ExampleVariable_Name() {
	v := variable.New(1, 2, 3, 4)
	v.Name = "v"
	fmt.Println(v)

	// Output:
	// v[4]([1 2 3 4])
}

func ExampleVariable_String() {
	v := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	fmt.Println(v)

	// Output:
	// variable[2 3]([1 2 3 4 5 6])
}

func ExampleZeroLike() {
	v := variable.New(1, 2, 3, 4)
	fmt.Println(variable.ZeroLike(v))

	// Output:
	// variable[4]([0 0 0 0])
}

func ExampleOneLike() {
	v := variable.New(1, 2, 3, 4)
	fmt.Println(variable.OneLike(v))

	// Output:
	// variable[4]([1 1 1 1])
}

func ExampleZeros() {
	fmt.Println(variable.Zeros(2, 3))

	// Output:
	// variable[2 3]([0 0 0 0 0 0])
}

func ExampleRand() {
	s := rand.Const()
	v := variable.Rand([]int{2, 3}, s)

	for _, row := range v.Data.Seq2() {
		fmt.Println(row)
	}

	// Output:
	// [0.9999275824802834 0.8856419373528862 0.38147752771154886]
	// [0.4812673234167829 0.44417259544314847 0.5210016660132573]
}

func ExampleRandn() {
	s := rand.Const()
	v := variable.Randn([]int{2, 3}, s)

	for _, row := range v.Data.Seq2() {
		fmt.Println(row)
	}

	// Output:
	// [0.5665360716030388 -0.6123972949371448 0.5898947122637695]
	// [-0.3678242340302933 1.0919575041640825 -0.4438344619606553]
}

func ExampleVariable_NumDims() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	fmt.Println(x.NumDims())

	// Output:
	// 2
}

func ExampleVariable_Size() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	fmt.Println(x.Size())

	// Output:
	// 6
}

func ExampleVariable_Unchain() {
	x := variable.New(1.0)

	y := variable.Pow(2.0)(x)
	fmt.Println(y.Creator) // Pow

	y.Unchain()
	fmt.Println(y.Creator) // nil

	// Output:
	// *variable.PowT[variable(1)]
	// <nil>
}

func ExampleVariable_UnchainBackward() {
	x := variable.New(1.0)

	y := variable.Pow(2.0)(x)
	z := variable.Sin(y)
	fmt.Println(y.Creator) // Pow
	fmt.Println(z.Creator) // Sin

	z.UnchainBackward()
	fmt.Println(y.Creator) // nil
	fmt.Println(z.Creator) // Sin

	z.Unchain()
	z.UnchainBackward()
	fmt.Println(y.Creator) // nil
	fmt.Println(z.Creator) // nil

	// Output:
	// *variable.PowT[variable(1)]
	// *variable.SinT[variable(1)]
	// <nil>
	// *variable.SinT[variable(1)]
	// <nil>
	// <nil>
}

func ExampleVariable_Backward() {
	x := variable.New(1.0)
	x.Backward()
	fmt.Println(x.Grad)

	x.Cleargrad()
	x.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable(1)
	// variable(1)
}

func Example_add() {
	fmt.Println(variable.AddGrad(nil, variable.New(1)))
	fmt.Println(variable.AddGrad(variable.New(1), variable.New(2)))

	// Output:
	// variable(1)
	// variable(3)
}

func Example_zip() {
	xs := []*variable.Variable{
		variable.New(1),
		variable.New(2),
	}
	gxs := []*variable.Variable{
		variable.New(1),
	}
	xs, gxs = variable.Zip(xs, gxs)
	fmt.Println(len(xs), len(gxs))

	// Output:
	// 1 1
}
