package autograd_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/model"
	"github.com/itsubaki/autograd/numerical"
	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/rand"
	"github.com/itsubaki/autograd/variable"
)

func Example() {
	x := variable.New(0.5)
	y := F.Square(F.Exp(F.Square(x)))
	y.Backward()

	fmt.Println(x.Grad)

	// Output:
	// variable([3.297442541400256])
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

	fmt.Printf("%.4f\n", numerical.Diff(f, v).Data.At(0, 0))

	// Output:
	// 3.2974
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
	fmt.Println(y.Creator.Input[0] == b)
	fmt.Println(y.Creator.Input[0].Creator)
	fmt.Println(y.Creator.Input[0].Creator.Input[0] == a)
	fmt.Println(y.Creator.Input[0].Creator.Input[0].Creator)
	fmt.Println(y.Creator.Input[0].Creator.Input[0].Creator.Input[0] == x)

	// Output:
	// variable([0.5])
	// variable([1.648721270700128])
	// variable([3.297442541400256])
	//
	// *variable.SquareT[variable([1.2840254166877414])]
	// true
	// *variable.ExpT[variable([0.25])]
	// true
	// *variable.SquareT[variable([0.5])]
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
	// variable([3.297442541400256])
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
	// variable([13])
	// variable([4]) variable([6])
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
	// variable([3])
}

func Example_inplace() {
	// p503
	x := variable.New(3.0)
	y := F.Add(x, x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable([6])
	// variable([2])
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
	// variable([2])
	// variable([3])
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
	// variable([32])
	// variable([64])
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

	fmt.Println(x.Grad, y.Grad)

	// Output:
	// variable([2]) variable([2])
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
	// variable([0.040000000000000036]) variable([0.040000000000000036])
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
	// variable([-2]) variable([400])
}

func Example_gradientDescent() {
	// p206
	rosenbrock := func(x0, x1 *variable.Variable) *variable.Variable {
		// 100 * (x1 - x0^2)^2 + (x0 - 1)^2
		y0 := F.Pow(2.0)(F.Sub(x1, F.Pow(2.0)(x0)))
		y1 := F.Pow(2.0)(F.AddC(-1.0, x0))
		return F.Add(F.MulC(100, y0), y1)
	}

	update := func(lr float64, x ...*variable.Variable) {
		for _, v := range x {
			v.Data = matrix.F2(v.Data, v.Grad.Data, func(a, b float64) float64 {
				return a - lr*b
			})
		}
	}

	x0 := variable.New(0.0)
	x1 := variable.New(2.0)

	lr := 0.001
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
	// variable([0]) variable([2])
	// variable([0.6837118569138317]) variable([0.4659526837427042])
	// variable([0.8263177857050957]) variable([0.6820311873361097])
	// variable([0.8947837494333546]) variable([0.8001896451930564])
	// variable([0.9334871723401226]) variable([0.8711213202579401])
	// variable([0.9569899983530249]) variable([0.9156532462021957])
	// variable([0.9718168065095137]) variable([0.9443132014542008])
	// variable([0.9813809710644894]) variable([0.9630332658658076])
	// variable([0.9876355102559093]) variable([0.9753740541653942])
	// variable([0.9917613994572028]) variable([0.9835575421346807])
	// variable([0.9944984367782456]) variable([0.9890050527419593])
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

		y := f(x)
		x.Cleargrad()
		y.Backward()

		x.Data = matrix.Sub(x.Data, matrix.Div(x.Grad.Data, gx2(x).Data))
	}

	// Output:
	// variable([2])
	// variable([1.4545454545454546])
	// variable([1.1510467893775467])
	// variable([1.0253259289766978])
	// variable([1.0009084519430513])
	// variable([1.0000012353089454])
	// variable([1.000000000002289])
	// variable([1])
	// variable([1])
	// variable([1])
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

	for i := 0; i < iter; i++ {
		fmt.Println(x)

		y := f(x)
		x.Cleargrad()
		y.Backward(variable.Opts{CreateGraph: true})

		gx := x.Grad
		x.Cleargrad()
		gx.Backward()
		gx2 := x.Grad

		x.Data = matrix.Sub(x.Data, matrix.Div(gx.Data, gx2.Data))
	}

	// Output:
	// variable([2])
	// variable([1.4545454545454546])
	// variable([1.1510467893775467])
	// variable([1.0253259289766978])
	// variable([1.0009084519430513])
	// variable([1.0000012353089454])
	// variable([1.000000000002289])
	// variable([1])
	// variable([1])
	// variable([1])
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
	// variable([100])
}

func Example_linearRegression() {
	// p318
	s := rand.Const()
	xrand := matrix.Rand(100, 1, s)
	yrand := matrix.Rand(100, 1, s)

	// variable
	x := variable.NewFrom(xrand)                                                    // x = xrand
	t := variable.NewFrom(matrix.Add(matrix.MulC(2, xrand), matrix.AddC(5, yrand))) // t = 2x+5+yrand

	// parameter
	w := variable.New(0.0)
	b := variable.New(0.0)

	predict := func(x *variable.Variable) *variable.Variable {
		return F.Add(F.MatMul(x, w), b) // y = x.w + b
	}

	update := func(lr float64, x ...*variable.Variable) {
		for _, v := range x {
			v.Data = matrix.F2(v.Data, v.Grad.Data, func(a, b float64) float64 {
				return a - lr*b
			})
		}
	}

	lr := 0.1
	iters := 100
	var loss *variable.Variable

	for i := 0; i < iters; i++ {
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
	fmt.Println(w, b, loss)

	// Output:
	// w([2.185928791917266]) b([5.354857189898866]) loss([0.07384886041118585])
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

	x := variable.Rand(100, 1, s)
	t := variable.Rand(100, 1, s)

	for i := range 100 {
		y := m.Forward(x)
		loss := F.MeanSquaredError(y, t)

		m.Cleargrads()
		loss.Backward()
		o.Update(m)

		if i%10 == 0 {
			fmt.Printf("%.8f\n", loss.Data.At(0, 0))
		}
	}

	// Output:
	// 0.17547970
	// 0.07741569
	// 0.07284688
	// 0.07090371
	// 0.07005858
	// 0.06968885
	// 0.06952675
	// 0.06945531
	// 0.06942337
	// 0.06940859
}
