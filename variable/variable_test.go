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

func ExampleVariable_At() {
	v := variable.New(1, 2, 3, 4)
	fmt.Println(v.At(0, 0))
	fmt.Println(v.At(0, 1))
	fmt.Println(v.At(0, 2))
	fmt.Println(v.At(0, 3))

	// Output:
	// 1
	// 2
	// 3
	// 4
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
	s := rand.Const()
	for _, r := range variable.Rand(2, 3, s).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [0.9999275824802834 0.8856419373528862 0.38147752771154886]
	// [0.4812673234167829 0.44417259544314847 0.5210016660132573]
}

func ExampleRandn() {
	s := rand.Const()
	for _, r := range variable.Randn(2, 3, s).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [0.5665360716030388 -0.6123972949371448 0.5898947122637695]
	// [-0.3678242340302933 1.0919575041640825 -0.4438344619606553]
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
