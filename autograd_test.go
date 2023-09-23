package autograd_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/numerical"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Example() {
	x := variable.New(0.5)
	y := F.Square(F.Exp(F.Square(x)))
	y.Backward()

	fmt.Println(x.Grad)

	// Output:
	// variable[3.297442541400256]
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
	// variable[3.2974426293330694]
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
	// variable[0.5]
	// variable[1.648721270700128]
	// variable[3.297442541400256]
	//
	// *function.SquareT[variable[1.2840254166877414]]
	// true
	// *function.ExpT[variable[0.25]]
	// true
	// *function.SquareT[variable[0.5]]
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
	// variable[3.297442541400256]
}

func Example_add() {
	// p85
	x := variable.New(2.0)
	y := variable.New(3.0)
	z := F.Add(F.Square(x), F.Square(y))
	z.Backward()

	fmt.Println(z)
	fmt.Println(x.Grad, y.Grad)

	// Output:
	// variable[13]
	// variable[4] variable[6]
}

func Example_reuse() {
	// p90
	x := variable.New(3.0)
	y := F.Add(F.Add(x, x), x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[9]
	// variable[3]
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
	// variable[2]
	// variable[3]
}

func Example_generation() {
	// p107
	x := variable.New(2.0)
	a := F.Square(x)
	y := F.Add(F.Square(a), F.Square(a))
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[32]
	// variable[64]
}

func Example_shpere() {
	// p167
	shpere := func(x, y *variable.Variable) *variable.Variable {
		// x^2 + y^2
		return F.Add(F.Pow(2.0)(x), F.Pow(2.0)(y))
	}

	x := variable.New(1.0)
	y := variable.New(1.0)
	z := shpere(x, y)
	z.Backward()

	fmt.Println(x.Grad, y.Grad)

	// Output:
	// variable[2] variable[2]
}

func Example_matyas() {
	// p167
	matyas := func(x, y *variable.Variable) *variable.Variable {
		// 0.26(x^2 + y^2) - 0.48xy
		z0 := F.MulC(0.26, F.Add(F.Pow(2.0)(x), F.Pow(2.0)(y)))
		z1 := F.MulC(0.48, F.Mul(x, y))
		return F.Sub(z0, z1)
	}

	x := variable.New(1.0)
	y := variable.New(1.0)
	z := matyas(x, y)
	z.Backward()

	fmt.Println(x.Grad, y.Grad)

	// Output:
	// variable[0.040000000000000036] variable[0.040000000000000036]
}

func Example_rosenbrock() {
	// p205
	rosenbrock := func(x0, x1 *variable.Variable) *variable.Variable {
		// 100 * (x1 - x0^2)^2 + (x0 - 1)^2
		y0 := F.MulC(100, F.Pow(2.0)(F.Sub(x1, F.Pow(2.0)(x0))))
		y1 := F.Pow(2.0)(F.AddC(-1.0, x0))
		return F.Add(y0, y1)
	}

	x0 := variable.New(0.0)
	x1 := variable.New(2.0)
	y := rosenbrock(x0, x1)
	y.Backward()

	fmt.Println(x0.Grad, x1.Grad)

	// Output:
	// variable[-2] variable[400]
}

func Example_gradientDescent() {
	// p206
	rosenbrock := func(x0, x1 *variable.Variable) *variable.Variable {
		// 100 * (x1 - x0^2)^2 + (x0 - 1)^2
		y0 := F.MulC(100, F.Pow(2.0)(F.Sub(x1, F.Pow(2.0)(x0))))
		y1 := F.Pow(2.0)(F.AddC(-1.0, x0))
		return F.Add(y0, y1)
	}

	gd := func(lr float64) func(a, b float64) float64 {
		return func(a, b float64) float64 { return a - lr*b }
	}

	update := func(lr float64, v ...*variable.Variable) {
		for _, w := range v {
			w.Data = vector.F2(w.Data, w.Grad.Data, gd(lr))
		}
	}

	x0 := variable.New(0.0)
	x1 := variable.New(2.0)

	lr := 0.001
	iters := 10000

	for i := 0; i < iters+1; i++ {
		if i%1000 == 0 {
			fmt.Println(x0, x1)
		}

		x0.Cleargrad()
		x1.Cleargrad()
		y := rosenbrock(x0, x1)
		y.Backward()

		update(lr, x0, x1)
	}

	// Output:
	// variable[0] variable[2]
	// variable[0.6837118569138317] variable[0.4659526837427042]
	// variable[0.8263177857050957] variable[0.6820311873361097]
	// variable[0.8947837494333546] variable[0.8001896451930564]
	// variable[0.9334871723401226] variable[0.8711213202579401]
	// variable[0.9569899983530249] variable[0.9156532462021957]
	// variable[0.9718168065095137] variable[0.9443132014542008]
	// variable[0.9813809710644894] variable[0.9630332658658076]
	// variable[0.9876355102559093] variable[0.9753740541653942]
	// variable[0.9917613994572028] variable[0.9835575421346807]
	// variable[0.9944984367782456] variable[0.9890050527419593]
}

func Example_newton() {
	// p214
	f := func(x *variable.Variable) *variable.Variable {
		// y = x^4 - 2x^2
		y0 := F.Pow(4.0)(x)  // x^4
		y1 := F.Pow(2.0)(x)  // x^2
		y2 := F.MulC(2, y1)  // 2x^2
		return F.Sub(y0, y2) // x^4 - 2x^2
	}

	gx2 := func(x *variable.Variable) *variable.Variable {
		// y = 12x^2 - 4
		return F.AddC(-4.0, F.MulC(12, F.Pow(2.0)(x)))
	}

	x := variable.New(2.0)
	iter := 10

	for i := 0; i < iter; i++ {
		fmt.Println(x)

		x.Cleargrad()
		y := f(x)
		y.Backward()

		x.Data = vector.Sub(x.Data, vector.Div(x.Grad.Data, gx2(x).Data))
	}

	// Output:
	// variable[2]
	// variable[1.4545454545454546]
	// variable[1.1510467893775467]
	// variable[1.0253259289766978]
	// variable[1.0009084519430513]
	// variable[1.0000012353089454]
	// variable[1.000000000002289]
	// variable[1]
	// variable[1]
	// variable[1]
}

func Example_double() {
	// p258
	x := variable.New(2.0)
	y := F.Pow(2.0)(x)
	y.Backward()
	gx := x.Grad

	z := F.Add(F.Pow(3.0)(gx), y)
	x.Cleargrad()
	y.Cleargrad()
	z.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[100]
}
