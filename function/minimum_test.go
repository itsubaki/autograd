package function_test

import (
	"fmt"
	"testing"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func ExampleMinimum() {
	x0 := variable.New(
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
	)

	x1 := variable.New(
		0.3, 0.2, 0.1,
		0.6, 0.5, 0.4,
	)

	y := F.Minimum(x0, x1)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x0.Grad)
	fmt.Println(x1.Grad)

	// Output:
	// variable[6]([0.1 0.2 0.1 0.4 0.5 0.4])
	// variable[6]([1 1 0 1 1 0])
	// variable[6]([0 0 1 0 0 1])
}

func ExampleMinimum_broadcast() {
	x0 := variable.New(2)
	x1 := variable.New(1, 3, 4)

	y := F.Minimum(x0, x1)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x0.Grad)
	fmt.Println(x1.Grad)

	// Output:
	// variable[3]([1 2 2])
	// variable(2)
	// variable[3]([1 0 0])
}

func TestMinimum_broadcast(t *testing.T) {
	cases := []struct {
		name    string
		x0      *tensor.Tensor[float32]
		x1      *tensor.Tensor[float32]
		gy      *tensor.Tensor[float32]
		want    *tensor.Tensor[float32]
		wantGX0 *tensor.Tensor[float32]
		wantGX1 *tensor.Tensor[float32]
	}{
		{
			name:    "same shape with ties",
			x0:      tensor.New([]int{3}, []float32{1, 2, 3}),
			x1:      tensor.New([]int{3}, []float32{2, 2, 1}),
			gy:      tensor.New([]int{3}, []float32{2, 3, 5}),
			want:    tensor.New([]int{3}, []float32{1, 2, 1}),
			wantGX0: tensor.New([]int{3}, []float32{2, 3, 0}),
			wantGX1: tensor.New([]int{3}, []float32{0, 0, 5}),
		},
		{
			name:    "scalar on left",
			x0:      tensor.Scalar(float32(2)),
			x1:      tensor.New([]int{3}, []float32{1, 3, 4}),
			gy:      tensor.New([]int{3}, []float32{2, 3, 5}),
			want:    tensor.New([]int{3}, []float32{1, 2, 2}),
			wantGX0: tensor.Scalar(float32(8)),
			wantGX1: tensor.New([]int{3}, []float32{2, 0, 0}),
		},
		{
			name:    "scalar on right",
			x0:      tensor.New([]int{3}, []float32{1, 3, 4}),
			x1:      tensor.Scalar(float32(2)),
			gy:      tensor.New([]int{3}, []float32{2, 3, 5}),
			want:    tensor.New([]int{3}, []float32{1, 2, 2}),
			wantGX0: tensor.New([]int{3}, []float32{2, 0, 0}),
			wantGX1: tensor.Scalar(float32(8)),
		},
		{
			name:    "singleton dimension",
			x0:      tensor.New([]int{1}, []float32{2}),
			x1:      tensor.New([]int{3}, []float32{1, 3, 4}),
			gy:      tensor.New([]int{3}, []float32{2, 3, 5}),
			want:    tensor.New([]int{3}, []float32{1, 2, 2}),
			wantGX0: tensor.New([]int{1}, []float32{8}),
			wantGX1: tensor.New([]int{3}, []float32{2, 0, 0}),
		},
		{
			name:    "row broadcast across matrix",
			x0:      tensor.New([]int{2, 3}, []float32{1, 5, 3, 4, 2, 6}),
			x1:      tensor.New([]int{3}, []float32{2, 3, 4}),
			gy:      tensor.New([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
			want:    tensor.New([]int{2, 3}, []float32{1, 3, 3, 2, 2, 4}),
			wantGX0: tensor.New([]int{2, 3}, []float32{1, 0, 3, 0, 5, 0}),
			wantGX1: tensor.New([]int{3}, []float32{4, 2, 6}),
		},
		{
			name:    "both inputs broadcast",
			x0:      tensor.New([]int{2, 1}, []float32{1, 4}),
			x1:      tensor.New([]int{1, 3}, []float32{2, 3, 5}),
			gy:      tensor.New([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
			want:    tensor.New([]int{2, 3}, []float32{1, 1, 1, 2, 3, 4}),
			wantGX0: tensor.New([]int{2, 1}, []float32{6, 6}),
			wantGX1: tensor.New([]int{1, 3}, []float32{4, 5, 0}),
		},
		{
			name:    "noncontiguous matrix",
			x0:      tensor.Transpose(tensor.New([]int{3, 2}, []float32{1, 4, 5, 2, 3, 6})),
			x1:      tensor.New([]int{3}, []float32{2, 3, 4}),
			gy:      tensor.Transpose(tensor.New([]int{3, 2}, []float32{1, 4, 2, 5, 3, 6})),
			want:    tensor.New([]int{2, 3}, []float32{1, 3, 3, 2, 2, 4}),
			wantGX0: tensor.New([]int{2, 3}, []float32{1, 0, 3, 0, 5, 0}),
			wantGX1: tensor.New([]int{3}, []float32{4, 2, 6}),
		},
	}

	for _, c := range cases {
		x0, x1 := variable.From(c.x0), variable.From(c.x1)

		y := F.Minimum(x0, x1)
		if !tensor.SliceEqual(y.Shape(), c.want.Shape) || !tensor.IsCloseAll(y.Data, c.want) {
			t.Fatalf("%s: y = %v, want %v", c.name, y, variable.From(c.want))
		}

		y.Grad = variable.From(c.gy)
		y.Backward()

		if x0.Grad == nil || !tensor.SliceEqual(x0.Grad.Shape(), c.wantGX0.Shape) || !tensor.IsCloseAll(x0.Grad.Data, c.wantGX0) {
			t.Errorf("%s: x0.grad = %v, want %v", c.name, x0.Grad, variable.From(c.wantGX0))
		}

		if x1.Grad == nil || !tensor.SliceEqual(x1.Grad.Shape(), c.wantGX1.Shape) || !tensor.IsCloseAll(x1.Grad.Data, c.wantGX1) {
			t.Errorf("%s: x1.grad = %v, want %v", c.name, x1.Grad, variable.From(c.wantGX1))
		}
	}
}
