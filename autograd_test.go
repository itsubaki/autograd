package autograd_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/numerical"
	"github.com/itsubaki/autograd/variable"
)

func Example() {
	x := variable.New(0.5)
	y := F.Square(F.Exp(F.Square(x)))

	y.Backward()
	fmt.Println(x.Grad)

	// Output:
	// [3.297442541400256]
}

func Example_numericalDiff() {
	// p23
	v := []*variable.Variable{variable.New(0.5)}
	f := func(x ...*variable.Variable) *variable.Variable {
		A := F.Square
		B := F.Exp
		C := F.Square
		return C(B(A(x...)))
	}

	fmt.Println(numerical.Diff(f, v))

	// Output:
	// variable([3.2974426293330694])
}

func Example_creator() {
	// p40
	x := variable.New(0.5)

	a := F.Square(x)
	b := F.Exp(a)
	y := F.Square(b)

	y.Backward()

	fmt.Println(x)
	fmt.Println(y)
	fmt.Println(x.Grad)
	fmt.Println()

	// p40
	fmt.Println(y.Creator)
	fmt.Println(y.Creator.Input()[0] == b)
	fmt.Println(y.Creator.Input()[0].Creator)
	fmt.Println(y.Creator.Input()[0].Creator.Input()[0] == a)
	fmt.Println(y.Creator.Input()[0].Creator.Input()[0].Creator)
	fmt.Println(y.Creator.Input()[0].Creator.Input()[0].Creator.Input()[0] == x)

	// Output:
	// variable([0.5])
	// variable([1.648721270700128])
	// [3.297442541400256]
	//
	// *function.SquareT([[1.2840254166877414]])
	// true
	// *function.ExpT([[0.25]])
	// true
	// *function.SquareT([[0.5]])
	// true
}

func Example_func() {
	// p44
	x := variable.New(0.5)
	a := F.Square(x)
	b := F.Exp(a)
	y := F.Square(b)

	y.Backward()
	fmt.Println(x.Grad)

	// Output:
	// [3.297442541400256]
}

func Example_add() {
	// p85
	x := variable.New(2.0)
	y := variable.New(3.0)
	z := F.Add(F.Square(x), F.Square(y))

	z.Backward()
	fmt.Println(z)
	fmt.Println(x.Grad)
	fmt.Println(y.Grad)

	// Output:
	// variable([13])
	// [4]
	// [6]
}

func Example_reuse() {
	// p90
	x := variable.New(3.0)
	y := F.Add(F.Add(x, x), x)

	y.Backward()
	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable([9])
	// [3]
}

func Example_cleargrad() {
	// p92
	x := variable.New(3.0)
	y := F.Add(x, x)
	y.Backward()
	fmt.Println(x.Grad)

	x.Cleargrad()
	y = F.Add(F.Add(x, x), x)
	y.Backward()
	fmt.Println(x.Grad)

	// Output:
	// [2]
	// [3]
}

func Example_generation() {
	// p107
	x := variable.New(2.0)
	a := F.Square(x)
	y := F.Add(F.Square(a), F.Square(a))
	y.Backward(true)

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable([32])
	// [64]
}

func Example_retain() {
	// p122
	x0 := variable.New(1.0)
	x1 := variable.New(1.0)
	t := F.Add(x0, x1)
	y := F.Add(x0, t)
	y.Backward(true)

	fmt.Println(y.Grad, t.Grad)
	fmt.Println(x0.Grad, x1.Grad)
	fmt.Println()

	x0.Cleargrad()
	x1.Cleargrad()
	t = F.Add(x0, x1)
	y = F.Add(x0, t)
	y.Backward()

	fmt.Println(y.Grad, t.Grad)
	fmt.Println(x0.Grad, x1.Grad)

	// Output:
	// [1] [1]
	// [2] [1]
	//
	// [] []
	// [2] [1]
}

func Example_rosenbrock() {
	// p208
	rosenbrock := func(x ...*variable.Variable) *variable.Variable {
		// 100 * (x1 - x0^2)^2 + (x0 - 1)^2
		y0 := F.Pow(2.0)(F.Sub(x[1], F.Pow(2.0)(x[0])[0]))[0] // y0 = (x1 - x0^2)^2
		y1 := F.Mul(variable.NewLikeWith(100, y0), y0)        // 100 * (x1 - x0^2)^2
		y2 := F.Sub(x[0], variable.OneLike(y0))               // x0 - 1
		y3 := F.Pow(2.0)(y2)[0]                               // (x0 - 1)^2
		return F.Add(y1, y3)                                  // 100 * (x1 - x0^2)^2 + (x0 - 1)^2
	}

	x0 := variable.New(0.0)
	x1 := variable.New(2.0)
	y := rosenbrock(x0, x1)
	y.Backward()

	fmt.Println(x0.Grad)
	fmt.Println(x1.Grad)

	// Output:
	// [-2]
	// [400]
}
