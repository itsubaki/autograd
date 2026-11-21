package variable_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func ExampleConcat() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.New(
		7, 8,
		9, 10,
	).Reshape(2, 2)

	axis := 1
	z := variable.Concat(axis)(x, y)
	z.Backward()

	fmt.Println(z)
	fmt.Println(x.Grad)
	fmt.Println(y.Grad)

	// Output:
	// variable[2 5]([1 2 3 7 8 4 5 6 9 10])
	// variable[2 3]([1 1 1 1 1 1])
	// variable[2 2]([1 1 1 1])
}

func ExampleConcat_axis0() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.New(
		7, 8, 9,
	).Reshape(1, 3)

	axis := 0
	z := variable.Concat(axis)(x, y)
	z.Backward()

	fmt.Println(z)
	fmt.Println(x.Grad)
	fmt.Println(y.Grad)

	// Output:
	// variable[3 3]([1 2 3 4 5 6 7 8 9])
	// variable[2 3]([1 1 1 1 1 1])
	// variable[1 3]([1 1 1])
}

func ExampleConcat_double() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.New(
		7, 8,
		9, 10,
	).Reshape(2, 2)

	axis := 1
	z := variable.Concat(axis)(x, y)
	z.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(z)
	fmt.Println(x.Grad)
	fmt.Println(y.Grad)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[2 5]([1 2 3 7 8 4 5 6 9 10])
	// variable[2 3]([1 1 1 1 1 1])
	// variable[2 2]([1 1 1 1])
	// <nil>
}

func ExampleConcat_negativeAxis() {
	x0 := variable.New(1, 2)
	x1 := variable.New(3, 4)

	y := variable.Concat(-1)(x0, x1)
	y.Grad = variable.New(1, 2, 3, 4)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x0.Grad)
	fmt.Println(x1.Grad)

	// Output:
	// variable[4]([1 2 3 4])
	// variable[2]([1 2])
	// variable[2]([3 4])
}

func TestConcat_negativeAxis(t *testing.T) {
	cases := []struct {
		name    string
		axis    int
		x0      *tensor.Tensor[float32]
		x1      *tensor.Tensor[float32]
		gy      *tensor.Tensor[float32]
		want    *tensor.Tensor[float32]
		wantGX0 *tensor.Tensor[float32]
		wantGX1 *tensor.Tensor[float32]
	}{
		{
			name:    "vector",
			axis:    -1,
			x0:      tensor.New([]int{2}, []float32{1, 2}),
			x1:      tensor.New([]int{3}, []float32{3, 4, 5}),
			gy:      tensor.New([]int{5}, []float32{5, 4, 3, 2, 1}),
			want:    tensor.New([]int{5}, []float32{1, 2, 3, 4, 5}),
			wantGX0: tensor.New([]int{2}, []float32{5, 4}),
			wantGX1: tensor.New([]int{3}, []float32{3, 2, 1}),
		},
		{
			name:    "matrix last axis",
			axis:    -1,
			x0:      tensor.New([]int{2, 2}, []float32{1, 2, 3, 4}),
			x1:      tensor.New([]int{2, 1}, []float32{5, 6}),
			gy:      tensor.New([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
			want:    tensor.New([]int{2, 3}, []float32{1, 2, 5, 3, 4, 6}),
			wantGX0: tensor.New([]int{2, 2}, []float32{1, 2, 4, 5}),
			wantGX1: tensor.New([]int{2, 1}, []float32{3, 6}),
		},
		{
			name:    "matrix first axis",
			axis:    -2,
			x0:      tensor.New([]int{1, 2}, []float32{1, 2}),
			x1:      tensor.New([]int{2, 2}, []float32{3, 4, 5, 6}),
			gy:      tensor.New([]int{3, 2}, []float32{6, 5, 4, 3, 2, 1}),
			want:    tensor.New([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6}),
			wantGX0: tensor.New([]int{1, 2}, []float32{6, 5}),
			wantGX1: tensor.New([]int{2, 2}, []float32{4, 3, 2, 1}),
		},
		{
			name:    "three dimensional middle axis",
			axis:    -2,
			x0:      tensor.New([]int{2, 1, 2}, []float32{1, 2, 3, 4}),
			x1:      tensor.New([]int{2, 2, 2}, []float32{5, 6, 7, 8, 9, 10, 11, 12}),
			gy:      tensor.New([]int{2, 3, 2}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
			want:    tensor.New([]int{2, 3, 2}, []float32{1, 2, 5, 6, 7, 8, 3, 4, 9, 10, 11, 12}),
			wantGX0: tensor.New([]int{2, 1, 2}, []float32{1, 2, 7, 8}),
			wantGX1: tensor.New([]int{2, 2, 2}, []float32{3, 4, 5, 6, 9, 10, 11, 12}),
		},
		{
			name:    "attention cache axis",
			axis:    -2,
			x0:      tensor.New([]int{1, 2, 1, 2}, []float32{1, 2, 3, 4}),
			x1:      tensor.New([]int{1, 2, 2, 2}, []float32{5, 6, 7, 8, 9, 10, 11, 12}),
			gy:      tensor.New([]int{1, 2, 3, 2}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
			want:    tensor.New([]int{1, 2, 3, 2}, []float32{1, 2, 5, 6, 7, 8, 3, 4, 9, 10, 11, 12}),
			wantGX0: tensor.New([]int{1, 2, 1, 2}, []float32{1, 2, 7, 8}),
			wantGX1: tensor.New([]int{1, 2, 2, 2}, []float32{3, 4, 5, 6, 9, 10, 11, 12}),
		},
		{
			name:    "three dimensional first axis",
			axis:    -3,
			x0:      tensor.New([]int{1, 1, 2}, []float32{1, 2}),
			x1:      tensor.New([]int{1, 1, 2}, []float32{3, 4}),
			gy:      tensor.New([]int{2, 1, 2}, []float32{1, 2, 3, 4}),
			want:    tensor.New([]int{2, 1, 2}, []float32{1, 2, 3, 4}),
			wantGX0: tensor.New([]int{1, 1, 2}, []float32{1, 2}),
			wantGX1: tensor.New([]int{1, 1, 2}, []float32{3, 4}),
		},
	}

	for _, c := range cases {
		for _, axis := range []int{c.axis, c.axis + c.x0.NumDims()} {
			x0, x1 := variable.From(c.x0), variable.From(c.x1)

			y := variable.Concat(axis)(x0, x1)
			if !tensor.SliceEqual(y.Shape(), c.want.Shape) || !tensor.IsCloseAll(y.Data, c.want) {
				t.Fatalf("%s, axis %d: y = %v, want %v", c.name, axis, y, variable.From(c.want))
			}

			y.Grad = variable.From(c.gy)
			y.Backward()

			if x0.Grad == nil || !tensor.SliceEqual(x0.Grad.Shape(), c.wantGX0.Shape) || !tensor.IsCloseAll(x0.Grad.Data, c.wantGX0) {
				t.Errorf("%s, axis %d: x0.grad = %v, want %v", c.name, axis, x0.Grad, variable.From(c.wantGX0))
			}

			if x1.Grad == nil || !tensor.SliceEqual(x1.Grad.Shape(), c.wantGX1.Shape) || !tensor.IsCloseAll(x1.Grad.Data, c.wantGX1) {
				t.Errorf("%s, axis %d: x1.grad = %v, want %v", c.name, axis, x1.Grad, variable.From(c.wantGX1))
			}
		}
	}
}

func TestConcat_negativeAxisDouble(t *testing.T) {
	cases := []struct {
		name string
		axis int
	}{
		{
			name: "last axis",
			axis: -1,
		},
		{
			name: "first axis",
			axis: -2,
		},
	}

	for _, c := range cases {
		x0 := variable.New(1, 2).Reshape(1, 2)
		x1 := variable.New(3, 4).Reshape(1, 2)

		y := variable.Concat(c.axis)(x0, x1)
		loss := variable.Sum()(variable.Square(y))
		loss.Backward(variable.Opts{
			CreateGraph: true,
		})

		gx0 := x0.Grad
		x0.Cleargrad()
		x1.Cleargrad()
		gx0.Backward()

		if x0.Grad == nil || !tensor.SliceEqual(x0.Grad.Shape(), []int{1, 2}) || !tensor.IsCloseAll(x0.Grad.Data, tensor.Full([]int{1, 2}, float32(2))) {
			t.Errorf("%s: x0.grad = %v, want [[2 2]]", c.name, x0.Grad)
		}

		if x1.Grad == nil || !tensor.SliceEqual(x1.Grad.Shape(), []int{1, 2}) || !tensor.IsCloseAll(x1.Grad.Data, tensor.Zeros[float32](1, 2)) {
			t.Errorf("%s: x1.grad = %v, want [[0 0]]", c.name, x1.Grad)
		}
	}
}
