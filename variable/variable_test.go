package variable_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/autograd/variable"
)

func ExampleVariable() {
	v := variable.New(1, 2, 3, 4)
	fmt.Println(v)

	// Output:
	// variable([1 2 3 4])
}

func ExampleVariable_Name() {
	v := variable.New(1, 2, 3, 4)
	v.Name = "v"
	fmt.Println(v)

	// Output:
	// v([1 2 3 4])
}

func ExampleVariable_Name_matrix() {
	v := variable.NewOf(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)
	fmt.Println(v)

	// Output:
	// variable([[1 2 3] [4 5 6]])
}

func ExampleConst() {
	fmt.Println(variable.Const(1))

	// Output:
	// const([1])
}

func ExampleZeroLike() {
	v := variable.New(1, 2, 3, 4)
	fmt.Println(variable.ZeroLike(v))

	// Output:
	// variable([0 0 0 0])
}

func ExampleOneLike() {
	v := variable.New(1, 2, 3, 4)
	fmt.Println(variable.OneLike(v))

	// Output:
	// variable([1 1 1 1])
}

func ExampleZero() {
	fmt.Println(variable.Zero(2, 3))

	// Output:
	// variable([[0 0 0] [0 0 0]])
}

func ExampleRand() {
	s := rand.NewSource(1)
	for _, r := range variable.Rand(2, 3, s).Data {
		fmt.Println(r)
	}

	// Output:
	// [0.6046602879796196 0.9405090880450124 0.6645600532184904]
	// [0.4377141871869802 0.4246374970712657 0.6868230728671094]
}

func ExampleRandn() {
	s := rand.NewSource(1)
	for _, r := range variable.Randn(2, 3, s).Data {
		fmt.Println(r)
	}

	// Output:
	// [-1.233758177597947 -0.12634751070237293 -0.5209945711531503]
	// [2.28571911769958 0.3228052526115799 0.5900672875996937]
}

func ExampleVariable_Unchain() {
	x := variable.New(1.0)

	y := variable.Pow(2.0)(x)
	fmt.Println(y.Creator) // Pow

	y.Unchain()
	fmt.Println(y.Creator) // nil

	// Output:
	// *variable.PowT[variable([1])]
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
	// *variable.PowT[variable([1])]
	// *variable.SinT[variable([1])]
	// <nil>
	// *variable.SinT[variable([1])]
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
	// variable([1])
	// variable([1])
}

func Example_add() {
	fmt.Println(variable.AddGrad(nil, variable.New(1)))
	fmt.Println(variable.AddGrad(variable.New(1), variable.New(2)))

	// Output:
	// variable([1])
	// variable([3])
}
