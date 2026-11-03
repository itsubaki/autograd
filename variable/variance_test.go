package variable_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func TestVarianceNegativeAxes(t *testing.T) {
	cases := []struct {
		name      string
		axes      []int
		y, gy, gx *variable.Variable
	}{
		{
			name: "last axis",
			axes: []int{-1},
			y:    variable.New(2.0/3, 2.0/3),
			gy:   variable.New(3, 6),
			gx:   variable.New(-2, 0, 2, -4, 0, 4).Reshape(2, 3),
		},
		{
			name: "first axis",
			axes: []int{-2},
			y:    variable.New(2.25, 2.25, 2.25),
			gy:   variable.New(2, 4, 6),
			gx:   variable.New(-3, -6, -9, 3, 6, 9).Reshape(2, 3),
		},
		{
			name: "all negative axes",
			axes: []int{-2, -1},
			y:    variable.From(tensor.Scalar(float32(35.0 / 12))),
			gy:   variable.From(tensor.Scalar(float32(12))),
			gx:   variable.New(-10, -6, -2, 2, 6, 10).Reshape(2, 3),
		},
		{
			name: "mixed axes",
			axes: []int{0, -1},
			y:    variable.From(tensor.Scalar(float32(35.0 / 12))),
			gy:   variable.From(tensor.Scalar(float32(12))),
			gx:   variable.New(-10, -6, -2, 2, 6, 10).Reshape(2, 3),
		},
	}
	for _, c := range cases {
		x := variable.New(1, 2, 3, 4, 5, 6).Reshape(2, 3)

		y := variable.Variance(c.axes...)(x)
		if !tensor.SliceEqual(y.Shape(), c.y.Shape()) || !tensor.IsCloseAll(y.Data, c.y.Data) {
			t.Errorf("y=%v, want=%v", y, c.y)
		}

		y.Grad = c.gy
		y.Backward()
		if x.Grad == nil {
			t.Fatal("missing input gradient")
		}

		if !tensor.SliceEqual(x.Grad.Shape(), c.gx.Shape()) || !tensor.IsCloseAll(x.Grad.Data, c.gx.Data) {
			t.Errorf("gx=%v, want=%v", x.Grad, c.gx)
		}
	}
}

func ExampleVariance() {
	x := variable.New(
		-1, 0, 1,
		-1, 0, 1,
	).Reshape(2, 3)

	y := variable.Variance()(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(variable.Clip(-0.333, 0.333)(x.Grad))

	// Output:
	// variable(0.6666667)
	// variable[2 3]([-0.333 0 0.333 -0.333 0 0.333])
}

func ExampleVariance_axis01() {
	x := variable.New(
		-1, 0, 1,
		-1, 0, 1,
	).Reshape(2, 3)

	y := variable.Variance(0, 1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(variable.Clip(-0.333, 0.333)(x.Grad))

	// Output:
	// variable(0.6666667)
	// variable[2 3]([-0.333 0 0.333 -0.333 0 0.333])
}

func ExampleVariance_axis0() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Variance(0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3]([2.25 2.25 2.25])
	// variable[2 3]([-1.5 -1.5 -1.5 1.5 1.5 1.5])
}

func ExampleVariance_axis1() {
	x := variable.New(
		-3, 0, 3,
		-3, 0, 3,
	).Reshape(2, 3)

	y := variable.Variance(1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2]([6 6])
	// variable[2 3]([-2 0 2 -2 0 2])
}

func ExampleVariance_double() {
	x := variable.New(
		-3, 0, 3,
		-3, 0, 3,
	).Reshape(2, 3)

	y := variable.Variance(1)(x)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)
	fmt.Println(x.Grad)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(x.Grad) // NOTE: zeros..., why?

	// Output:
	// variable[2]([6 6])
	// variable[2 3]([-2 0 2 -2 0 2])
	// variable[2 3]([0 0 0 0 0 0])
}
