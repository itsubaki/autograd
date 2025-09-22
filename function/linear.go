package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func Linear(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{
		Forwarder: &LinearT{},
	}).First(x...)
}

type LinearT struct {
	x, w, b *variable.Variable
}

func (f *LinearT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x, f.w = x[0], x[1]

	y := tensor.MatMul(x[0].Data, x[1].Data)
	if len(x) < 3 {
		// no bias
		return []*variable.Variable{
			variable.From(y),
		}
	}

	// add bias
	f.b, y = x[2], tensor.Add(y, x[2].Data)
	return []*variable.Variable{
		variable.From(y),
	}
}

func (f *LinearT) Backward(gy ...*variable.Variable) []*variable.Variable {
	axw := axes(f.w.Data.NumDims())
	gx := MatMul(gy[0], Transpose(axw...)(f.w)) // gy * w.T
	if !tensor.ShapeEqual(gx.Shape(), f.x.Shape()) {
		gx = SumTo(f.x.Shape()...)(gx)
	}

	axx := axes(f.x.Data.NumDims())
	gw := MatMul(Transpose(axx...)(f.x), gy[0]) // x.T * gy
	if !tensor.ShapeEqual(gw.Shape(), f.w.Shape()) {
		gw = SumTo(f.w.Shape()...)(gw)
	}

	if f.b != nil {
		// add bias
		gb := SumTo(f.b.Shape()...)(gy[0])
		return []*variable.Variable{
			gx,
			gw,
			gb,
		}
	}

	// no bias
	return []*variable.Variable{
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
