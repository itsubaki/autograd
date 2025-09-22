package variable

import "github.com/itsubaki/autograd/tensor"

func MatMul(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &MatMulT{},
	}).First(x...)
}

type MatMulT struct {
	x, w *Variable
}

func (f *MatMulT) Forward(x ...*Variable) []*Variable {
	f.x, f.w = x[0], x[1]

	y := tensor.MatMul(x[0].Data, x[1].Data)
	return []*Variable{
		From(y),
	}
}

func (f *MatMulT) Backward(gy ...*Variable) []*Variable {
	axw := axes(f.w.Data.NumDims())
	gx := MatMul(gy[0], Transpose(axw...)(f.w)) // gy * w.T
	if !equal(gx.Shape(), f.x.Shape()) {
		gx = SumTo(f.x.Shape()...)(gx)
	}

	axx := axes(f.x.Data.NumDims())
	gw := MatMul(Transpose(axx...)(f.x), gy[0]) // x.T * gy
	if !equal(gw.Shape(), f.w.Shape()) {
		gw = SumTo(f.w.Shape()...)(gw)
	}

	return []*Variable{
		gx,
		gw,
	}
}

func axes(ndim int) []int {
	axes := make([]int, ndim)
	for i := range ndim - 2 {
		axes[i] = i
	}

	// swap last two axes
	axes[ndim-2] = ndim - 1
	axes[ndim-1] = ndim - 2
	return axes
}
