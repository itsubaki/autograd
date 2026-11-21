package variable_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func ExampleMin() {
	x := variable.New(
		1, 2, 3,
		4, -5, 6,
	).Reshape(2, 3)

	y := variable.Min()(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(-5)
	// variable[2 3]([0 0 0 0 1 0])
}

func ExampleMin_axis01() {
	x := variable.New(
		1, 2, 3,
		4, -5, 6,
	).Reshape(2, 3)

	y := variable.Min(0, 1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(-5)
	// variable[2 3]([0 0 0 0 1 0])
}

func ExampleMin_axis0() {
	x := variable.New(
		1, 2, 3,
		4, -5, 6,
	).Reshape(2, 3)

	y := variable.Min(0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3]([1 -5 3])
	// variable[2 3]([1 0 1 0 1 0])
}

func ExampleMin_axis1() {
	x := variable.New(
		1, 2, 3,
		4, -5, 6,
	).Reshape(2, 3)

	y := variable.Min(1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2]([1 -5])
	// variable[2 3]([1 0 0 0 1 0])
}

func ExampleMin_axis21n() {
	x := variable.New(
		1, 2, 3,
		4, -5, 6,
	).Reshape(2, 3)

	y := variable.Min(-2, -1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(-5)
	// variable[2 3]([0 0 0 0 1 0])
}

func ExampleMin_double() {
	x := variable.New(
		1, 2, 3,
		4, -5, 6,
	).Reshape(2, 3)

	y := variable.Min()(x)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)
	fmt.Println(x.Grad)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable(-5)
	// variable[2 3]([0 0 0 0 1 0])
	// <nil>
}

func ExampleMin_closeValues() {
	x := variable.New(1, 1.000005)

	y := variable.Min()(x)
	y.Backward()

	fmt.Println(x.Grad)

	// Output:
	// variable[2]([1 0])
}

func TestMin_exactSelection(t *testing.T) {
	cases := []struct {
		name string
		x    *tensor.Tensor[float32]
		axes []int
		gy   *tensor.Tensor[float32]
		want *tensor.Tensor[float32]
		gx   *tensor.Tensor[float32]
	}{
		{
			name: "close positive values",
			x:    tensor.New([]int{2}, []float32{1, 1.000005}),
			axes: nil,
			gy:   tensor.Scalar(float32(3)),
			want: tensor.Scalar(float32(1)),
			gx:   tensor.New([]int{2}, []float32{3, 0}),
		},
		{
			name: "close negative values",
			x:    tensor.New([]int{2}, []float32{-1, -1.000005}),
			axes: nil,
			gy:   tensor.Scalar(float32(3)),
			want: tensor.Scalar(float32(-1.000005)),
			gx:   tensor.New([]int{2}, []float32{0, 3}),
		},
		{
			name: "close values near zero",
			x:    tensor.New([]int{2}, []float32{0, 5e-9}),
			axes: nil,
			gy:   tensor.Scalar(float32(3)),
			want: tensor.Scalar(float32(0)),
			gx:   tensor.New([]int{2}, []float32{3, 0}),
		},
		{
			name: "large close values",
			x:    tensor.New([]int{2}, []float32{1000000, 1000001}),
			axes: nil,
			gy:   tensor.Scalar(float32(3)),
			want: tensor.Scalar(float32(1000000)),
			gx:   tensor.New([]int{2}, []float32{3, 0}),
		},
		{
			name: "exact ties keep all matching gradients",
			x:    tensor.New([]int{3}, []float32{1, 1, 1.000005}),
			axes: nil,
			gy:   tensor.Scalar(float32(3)),
			want: tensor.Scalar(float32(1)),
			gx:   tensor.New([]int{3}, []float32{3, 3, 0}),
		},
		{
			name: "last axis",
			x:    tensor.New([]int{2, 2}, []float32{1, 1.000005, -1, -1.000005}),
			axes: []int{1},
			gy:   tensor.New([]int{2}, []float32{2, 3}),
			want: tensor.New([]int{2}, []float32{1, -1.000005}),
			gx:   tensor.New([]int{2, 2}, []float32{2, 0, 0, 3}),
		},
		{
			name: "negative last axis",
			x:    tensor.New([]int{2, 2}, []float32{1, 1.000005, -1, -1.000005}),
			axes: []int{-1},
			gy:   tensor.New([]int{2}, []float32{2, 3}),
			want: tensor.New([]int{2}, []float32{1, -1.000005}),
			gx:   tensor.New([]int{2, 2}, []float32{2, 0, 0, 3}),
		},
		{
			name: "first axis",
			x:    tensor.New([]int{2, 2}, []float32{1, -1, 1.000005, -1.000005}),
			axes: []int{0},
			gy:   tensor.New([]int{2}, []float32{2, 3}),
			want: tensor.New([]int{2}, []float32{1, -1.000005}),
			gx:   tensor.New([]int{2, 2}, []float32{2, 0, 0, 3}),
		},
		{
			name: "multiple axes",
			x:    tensor.New([]int{2, 2, 2}, []float32{1, 1.000005, 2, 2.000005, 1.000005, 1, 2.000005, 2}),
			axes: []int{0, 2},
			gy:   tensor.New([]int{2}, []float32{2, 3}),
			want: tensor.New([]int{2}, []float32{1, 2}),
			gx:   tensor.New([]int{2, 2, 2}, []float32{2, 0, 3, 0, 0, 2, 0, 3}),
		},
		{
			name: "noncontiguous input",
			x:    tensor.Transpose(tensor.New([]int{2, 2}, []float32{1, 1.000005, -1, -1.000005})),
			axes: []int{0},
			gy:   tensor.New([]int{2}, []float32{2, 3}),
			want: tensor.New([]int{2}, []float32{1, -1.000005}),
			gx:   tensor.New([]int{2, 2}, []float32{2, 0, 0, 3}),
		},
	}

	for _, c := range cases {
		x := variable.From(c.x)

		y := variable.Min(c.axes...)(x)
		if !tensor.SliceEqual(y.Shape(), c.want.Shape) || !tensor.IsCloseAll(y.Data, c.want, 0) {
			t.Fatalf("%s: y = %v, want %v", c.name, y, variable.From(c.want))
		}

		y.Grad = variable.From(c.gy)
		y.Backward()

		if x.Grad == nil || !tensor.SliceEqual(x.Grad.Shape(), c.gx.Shape) || !tensor.IsCloseAll(x.Grad.Data, c.gx, 0) {
			t.Errorf("%s: x.grad = %v, want %v", c.name, x.Grad, variable.From(c.gx))
		}
	}
}
