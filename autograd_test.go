package autograd_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/model"
	"github.com/itsubaki/autograd/numerical"
	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/rand"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func Example() {
	x := variable.New(0.5)
	y := F.Square(F.Exp(F.Square(x)))
	y.Backward()

	fmt.Println(x.Grad)

	// Output:
	// variable(3.2974427)
}

func Example_numericalDiff() {
	// p23
	v := variable.New(0.5)
	f := func(x ...*variable.Variable) *variable.Variable {
		A := F.Square
		B := F.Exp
		C := F.Square
		return C(B(A(x...)))
	}

	fmt.Printf("%.4f\n", numerical.Diff(f, v).At())

	// Output:
	// 3.2973
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
	fmt.Println(".")

	// p40
	fmt.Println(y.Creator)
	fmt.Println(y.Creator.Input[0] == b)
	fmt.Println(y.Creator.Input[0].Creator)
	fmt.Println(y.Creator.Input[0].Creator.Input[0] == a)
	fmt.Println(y.Creator.Input[0].Creator.Input[0].Creator)
	fmt.Println(y.Creator.Input[0].Creator.Input[0].Creator.Input[0] == x)

	// Output:
	// variable(0.5)
	// variable(1.6487213)
	// variable(3.2974427)
	// .
	// *variable.SquareT[variable(1.2840254)]
	// true
	// *variable.ExpT[variable(0.25)]
	// true
	// *variable.SquareT[variable(0.5)]
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
	// variable(3.2974427)
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
	// variable(13)
	// variable(4) variable(6)
}

func Example_reuse() {
	// p90
	x := variable.New(3.0)
	y := F.Add(F.Add(x, x), x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(9)
	// variable(3)
}

func Example_inplace() {
	// p503
	x := variable.New(3.0)
	y := F.Add(x, x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(6)
	// variable(2)
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
	// variable(2)
	// variable(3)
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
	// variable(32)
	// variable(64)
}

func Example_sphere() {
	// p167
	sphere := func(x, y *variable.Variable) *variable.Variable {
		// x^2 + y^2
		return F.Add(F.Pow(2.0)(x), F.Pow(2.0)(y))
	}

	x := variable.New(1.0)
	y := variable.New(1.0)
	z := sphere(x, y)
	z.Backward()

	fmt.Println(x.Grad)
	fmt.Println(y.Grad)

	// Output:
	// variable(2)
	// variable(2)
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

	fmt.Println(x.Grad)
	fmt.Println(y.Grad)

	// Output:
	// variable(0.03999999)
	// variable(0.03999999)
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

	fmt.Println(x0.Grad)
	fmt.Println(x1.Grad)

	// Output:
	// variable(-2)
	// variable(400)
}

func Example_gradientDescent() {
	// p206
	rosenbrock := func(x0, x1 *variable.Variable) *variable.Variable {
		// 100 * (x1 - x0^2)^2 + (x0 - 1)^2
		y0 := F.Pow(2.0)(F.Sub(x1, F.Pow(2.0)(x0)))
		y1 := F.Pow(2.0)(F.AddC(-1.0, x0))
		return F.Add(F.MulC(100, y0), y1)
	}

	update := func(lr float32, x ...*variable.Variable) {
		for _, v := range x {
			v.Data = tensor.F2(v.Data, v.Grad.Data, func(a, b float32) float32 {
				return a - lr*b
			})
		}
	}

	x0 := variable.New(0.0)
	x1 := variable.New(2.0)

	lr := float32(0.001)
	iters := 10000

	for i := range iters + 1 {
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
	// variable(0) variable(2)
	// variable(0.68371195) variable(0.4659528)
	// variable(0.82631814) variable(0.6820318)
	// variable(0.8947841) variable(0.8001903)
	// variable(0.93348724) variable(0.87112147)
	// variable(0.9569893) variable(0.9156519)
	// variable(0.97181624) variable(0.94431216)
	// variable(0.98138046) variable(0.9630323)
	// variable(0.9876351) variable(0.9753732)
	// variable(0.9917611) variable(0.98355687)
	// variable(0.99449813) variable(0.98900443)
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

	for range iter {
		fmt.Println(x)

		y := f(x)
		x.Cleargrad()
		y.Backward()

		x.Data = tensor.Sub(x.Data, tensor.Div(x.Grad.Data, gx2(x).Data))
	}

	// Output:
	// variable(2)
	// variable(1.4545455)
	// variable(1.1510468)
	// variable(1.0253259)
	// variable(1.0009084)
	// variable(1.0000012)
	// variable(1)
	// variable(1)
	// variable(1)
	// variable(1)
}

func Example_newton_double() {
	// p239
	f := func(x *variable.Variable) *variable.Variable {
		// y = x^4 - 2x^2
		y0 := F.Pow(4.0)(x)  // x^4
		y1 := F.Pow(2.0)(x)  // x^2
		y2 := F.MulC(2, y1)  // 2x^2
		return F.Sub(y0, y2) // x^4 - 2x^2
	}

	x := variable.New(2.0)
	iter := 10

	for range iter {
		fmt.Println(x)

		y := f(x)
		x.Cleargrad()
		y.Backward(variable.Opts{CreateGraph: true})

		gx := x.Grad
		x.Cleargrad()
		gx.Backward()
		gx2 := x.Grad

		x.Data = tensor.Sub(x.Data, tensor.Div(gx.Data, gx2.Data))
	}

	// Output:
	// variable(2)
	// variable(1.4545455)
	// variable(1.1510468)
	// variable(1.0253259)
	// variable(1.0009084)
	// variable(1.0000012)
	// variable(1)
	// variable(1)
	// variable(1)
	// variable(1)
}

func Example_double() {
	// p258
	// y = x^2
	// z = (dy/dx)^3 + y
	// dz/dx = d/dx(8x^3 + x^2) = 24x^2 + 2x
	x := variable.New(2.0)
	y := F.Pow(2.0)(x)
	y.Backward(variable.Opts{CreateGraph: true})
	gx := x.Grad

	z := F.Add(F.Pow(3.0)(gx), y)
	x.Cleargrad()
	z.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable(100)
}

func Example_linearRegression() {
	// p318
	s := rand.Const()
	xrand := tensor.Rand([]int{100, 1}, s)
	yrand := tensor.Rand([]int{100, 1}, s)

	// variable
	x := variable.From(xrand)                                                    // x = xrand
	t := variable.From(tensor.Add(tensor.MulC(2, xrand), tensor.AddC(5, yrand))) // t = 2x+5+yrand

	// parameter
	w := variable.New(0.0).Reshape(1, 1)
	b := variable.New(0.0).Reshape(1, 1)

	predict := func(x *variable.Variable) *variable.Variable {
		return F.Add(F.MatMul(x, w), b) // y = x.w + b
	}

	update := func(lr float32, x ...*variable.Variable) {
		for _, v := range x {
			v.Data = tensor.F2(v.Data, v.Grad.Data, func(a, b float32) float32 {
				return a - lr*b
			})
		}
	}

	lr := float32(0.1)
	iters := 100

	var loss *variable.Variable
	for range iters {
		y := predict(x)
		loss = F.MeanSquaredError(y, t)

		w.Cleargrad()
		b.Cleargrad()
		loss.Backward()

		update(lr, w, b)
	}

	w.Name = "w"
	b.Name = "b"
	loss.Name = "loss"

	fmt.Printf("%.4f\n", w.At())
	fmt.Printf("%.4f\n", b.At())
	fmt.Printf("%.4f\n", loss.At())

	// Output:
	// 2.2133
	// 5.3925
	// 0.0777
}

func Example_mlp() {
	s := rand.Const()
	m := model.NewMLP([]int{10, 1},
		model.WithMLPSource(s),
		model.WithMLPActivation(F.ReLU),
	)

	o := optimizer.SGD{
		LearningRate: 0.2,
	}

	x := variable.Rand([]int{100, 1}, s)
	t := variable.Rand([]int{100, 1}, s)

	for i := range 100 {
		y := m.Forward(x)
		loss := F.MeanSquaredError(y, t)

		m.Cleargrads()
		loss.Backward()
		o.Update(m.Params())

		if i%10 == 0 {
			fmt.Printf("%.6f\n", loss.At())
		}
	}

	// Output:
	// 0.212382
	// 0.081022
	// 0.077388
	// 0.075996
	// 0.075453
	// 0.075238
	// 0.075153
	// 0.075119
	// 0.075105
	// 0.075100
}
