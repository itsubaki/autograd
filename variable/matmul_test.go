package variable_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func ExampleMatMul() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	w := variable.New(
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	).Reshape(3, 4)

	y := variable.MatMul(x, w)
	y.Backward()

	fmt.Println(x.Grad.Shape(), x.Grad.Data.Data)
	fmt.Println(w.Grad.Shape(), w.Grad.Data.Data)

	// Output:
	// [2 3] [10 26 42 10 26 42]
	// [3 4] [5 5 5 5 7 7 7 7 9 9 9 9]
}

func TestMatMul(t *testing.T) {
	cases := []struct {
		x, w *variable.Variable
		y    *variable.Variable
		gx   *variable.Variable
		gw   *variable.Variable
	}{
		{
			x: variable.New(
				1, 2, 3,
				4, 5, 6,
			).Reshape(2, 3),
			w: variable.New(
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,
			).Reshape(3, 4),
			y: variable.New(
				38, 44, 50, 56,
				83, 98, 113, 128,
			).Reshape(2, 4),
			gx: variable.New(
				10, 26, 42,
				10, 26, 42,
			).Reshape(2, 3),
			gw: variable.New(
				5, 5, 5, 5,
				7, 7, 7, 7,
				9, 9, 9, 9,
			).Reshape(3, 4),
		},
		{
			x: variable.New(
				1, 2, 3,
				4, 5, 6,

				7, 8, 9,
				10, 11, 12,
			).Reshape(2, 2, 3),
			w: variable.New(
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,

				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
			).Reshape(2, 3, 4),
			y: variable.New(
				38, 44, 50, 56,
				83, 98, 113, 128,

				7, 8, 9, 0,
				10, 11, 12, 0,
			).Reshape(2, 2, 4),
			gx: variable.New(
				10, 26, 42,
				10, 26, 42,

				1, 1, 1,
				1, 1, 1,
			).Reshape(2, 2, 3),
			gw: variable.New(
				5, 5, 5, 5,
				7, 7, 7, 7,
				9, 9, 9, 9,

				17, 17, 17, 17,
				19, 19, 19, 19,
				21, 21, 21, 21,
			).Reshape(2, 3, 4),
		},
	}

	for _, c := range cases {
		y := variable.MatMul(c.x, c.w)
		if !tensor.IsCloseAll(y.Data, c.y.Data, 1e-8, 1e-5) {
			t.Errorf("got=%v, want=%v", y, c.y)
		}

		y.Backward()
		if !tensor.IsCloseAll(c.x.Grad.Data, c.gx.Data, 1e-8, 1e-5) {
			t.Errorf("got=%v, want=%v", c.x.Grad, c.gx)
		}

		if !tensor.IsCloseAll(c.w.Grad.Data, c.gw.Data, 1e-8, 1e-5) {
			t.Errorf("got=%v, want=%v", c.w.Grad, c.gw)
		}
	}
}
