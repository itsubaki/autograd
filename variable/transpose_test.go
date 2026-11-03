package variable_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func ExampleTranspose() {
	// p286
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Transpose(-1, -2)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3 2]([1 4 2 5 3 6])
	// variable[2 3]([1 1 1 1 1 1])
}

func ExampleTranspose_double() {
	// p286
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Transpose(-1, -2)(x)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)
	fmt.Println(x.Grad)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[3 2]([1 4 2 5 3 6])
	// variable[2 3]([1 1 1 1 1 1])
	// <nil>
}

func TestTranspose(t *testing.T) {
	cases := []struct {
		x    *variable.Variable
		axes []int
		y    *variable.Variable
		gx   *variable.Variable
	}{
		{
			x: variable.New(
				1, 2, 3,
				4, 5, 6,
			).Reshape(2, 3),
			axes: []int{1, 0},
			y: variable.New(
				1, 4,
				2, 5,
				3, 6,
			).Reshape(3, 2),
			gx: variable.New(
				1, 1, 1,
				1, 1, 1,
			).Reshape(2, 3),
		},
		{
			x: variable.New(
				1, 2, 3,
				4, 5, 6,

				7, 8, 9,
				10, 11, 12,
			).Reshape(2, 2, 3),
			axes: []int{0, 2, 1},
			y: variable.New(
				1, 4,
				2, 5,
				3, 6,

				7, 10,
				8, 11,
				9, 12,
			).Reshape(2, 3, 2),
			gx: variable.New(
				1, 1, 1,
				1, 1, 1,

				1, 1, 1,
				1, 1, 1,
			).Reshape(2, 2, 3),
		},
		{
			// transpose 2 0 1
			x: variable.New(
				1, 2, 3,
				4, 5, 6,

				7, 8, 9,
				10, 11, 12,
			).Reshape(2, 2, 3),
			axes: []int{2, 0, 1},
			y: variable.New(
				1, 4,
				7, 10,

				2, 5,
				8, 11,

				3, 6,
				9, 12,
			).Reshape(3, 2, 2),
			gx: variable.New(
				1, 1, 1,
				1, 1, 1,

				1, 1, 1,
				1, 1, 1,
			).Reshape(2, 2, 3),
		},
		{
			// transpose -1 -3 -2
			x: variable.New(
				1, 2, 3,
				4, 5, 6,

				7, 8, 9,
				10, 11, 12,
			).Reshape(2, 2, 3),
			axes: []int{-1, -3, -2},
			y: variable.New(
				1, 4,
				7, 10,

				2, 5,
				8, 11,

				3, 6,
				9, 12,
			).Reshape(3, 2, 2),
			gx: variable.New(
				1, 1, 1,
				1, 1, 1,

				1, 1, 1,
				1, 1, 1,
			).Reshape(2, 2, 3),
		},
	}

	for _, c := range cases {
		y := variable.Transpose(c.axes...)(c.x)
		if !tensor.IsCloseAll(y.Data, c.y.Data) {
			t.Errorf("got=%v, want=%v", y.Data, c.y.Data)
		}

		y.Backward()
		if !tensor.IsCloseAll(c.x.Grad.Data, c.gx.Data) {
			t.Errorf("got=%v, want=%v", c.x.Grad.Data, c.gx.Data)
		}
	}
}

func TestTransposeDefaultAxes(t *testing.T) {
	cases := []struct {
		name         string
		x, y, gy, gx *variable.Variable
	}{
		{
			name: "matrix",
			x:    variable.New(1, 2, 3, 4, 5, 6).Reshape(2, 3),
			y:    variable.New(1, 4, 2, 5, 3, 6).Reshape(3, 2),
			gy:   variable.New(1, 2, 3, 4, 5, 6).Reshape(3, 2),
			gx:   variable.New(1, 3, 5, 2, 4, 6).Reshape(2, 3),
		},
		{
			name: "three dimensions",
			x:    variable.New(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12).Reshape(2, 2, 3),
			y:    variable.New(1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12).Reshape(3, 2, 2),
			gy:   variable.New(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12).Reshape(3, 2, 2),
			gx:   variable.New(1, 5, 9, 3, 7, 11, 2, 6, 10, 4, 8, 12).Reshape(2, 2, 3),
		},
	}
	for _, c := range cases {
		y := variable.Transpose()(c.x)
		if !tensor.SliceEqual(y.Shape(), c.y.Shape()) || !tensor.IsCloseAll(y.Data, c.y.Data) {
			t.Errorf("y=%v, want=%v", y, c.y)
		}

		y.Grad = c.gy
		y.Backward()
		if c.x.Grad == nil {
			t.Fatal("missing input gradient")
		}

		if !tensor.SliceEqual(c.x.Grad.Shape(), c.gx.Shape()) || !tensor.IsCloseAll(c.x.Grad.Data, c.gx.Data) {
			t.Errorf("gx=%v, want=%v", c.x.Grad, c.gx)
		}
	}
}

func TestInvPerm_invalid(t *testing.T) {
	cases := []struct {
		ndim int
		axes []int
	}{
		{
			ndim: 3,
			axes: []int{-10, 0, 10},
		},
	}

	for _, c := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					return
				}

				t.Errorf("unexpected panic for axes %v", c.axes)
			}()

			_ = variable.InvPerm(c.ndim, c.axes...)
			t.Fail()
		}()
	}
}
