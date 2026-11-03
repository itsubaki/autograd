package variable_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func TestMeanNegativeAxes(t *testing.T) {
	cases := []struct {
		name      string
		axes      []int
		y, gy, gx *variable.Variable
	}{
		{
			name: "last axis",
			axes: []int{-1},
			y:    variable.New(2, 5),
			gy:   variable.New(3, 6),
			gx:   variable.New(1, 1, 1, 2, 2, 2).Reshape(2, 3),
		},
		{
			name: "first axis",
			axes: []int{-2},
			y:    variable.New(2.5, 3.5, 4.5),
			gy:   variable.New(2, 4, 6),
			gx:   variable.New(1, 2, 3, 1, 2, 3).Reshape(2, 3),
		},
		{
			name: "all negative axes",
			axes: []int{-2, -1},
			y:    variable.From(tensor.Scalar(float32(3.5))),
			gy:   variable.From(tensor.Scalar(float32(12))),
			gx:   variable.New(2, 2, 2, 2, 2, 2).Reshape(2, 3),
		},
		{
			name: "mixed axes",
			axes: []int{0, -1},
			y:    variable.From(tensor.Scalar(float32(3.5))),
			gy:   variable.From(tensor.Scalar(float32(12))),
			gx:   variable.New(2, 2, 2, 2, 2, 2).Reshape(2, 3),
		},
	}
	for _, c := range cases {
		x := variable.New(1, 2, 3, 4, 5, 6).Reshape(2, 3)

		y := variable.Mean(c.axes...)(x)
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

func ExampleMean() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Mean()(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(variable.Clip(0, 0.1666)(x.Grad))

	// Output:
	// variable(3.5)
	// variable[2 3]([0.1666 0.1666 0.1666 0.1666 0.1666 0.1666])
}

func ExampleMean_axis01() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Mean(0, 1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(variable.Clip(0, 0.1666)(x.Grad))

	// Output:
	// variable(3.5)
	// variable[2 3]([0.1666 0.1666 0.1666 0.1666 0.1666 0.1666])
}

func ExampleMean_axis0() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Mean(0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3]([2.5 3.5 4.5])
	// variable[2 3]([0.5 0.5 0.5 0.5 0.5 0.5])
}

func ExampleMean_axis1() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Mean(1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(variable.Clip(0, 0.3333)(x.Grad))

	// Output:
	// variable[2]([2 5])
	// variable[2 3]([0.3333 0.3333 0.3333 0.3333 0.3333 0.3333])
}

func ExampleMean_double() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Mean(0, 1)(x)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable(3.5)
	// <nil>
}
