package variable

import "github.com/itsubaki/autograd/vector"

func AddC(c float64, x ...*Variable) *Variable {
	return (&Function{Forwarder: &AddT{}}).ApplyAndFirst(Const(c), x[0])
}

func Add(x ...*Variable) *Variable {
	return (&Function{Forwarder: &AddT{}}).ApplyAndFirst(x...)
}

type AddT struct {
	x0Shape, x1Shape []int
}

func (f *AddT) Forward(x ...*Variable) []*Variable {
	f.x0Shape, f.x1Shape = vector.Shape(x[0].Data), vector.Shape(x[1].Data)

	x0, x1 := vector.Broadcast(x[0].Data, x[1].Data)
	y := vector.Add(x0, x1)
	return []*Variable{
		New(y...),
	}
}

func (f *AddT) Backward(gy ...*Variable) []*Variable {
	if vector.Equal(f.x0Shape, f.x1Shape) {
		return []*Variable{
			gy[0],
			gy[0],
		}
	}

	return []*Variable{
		SumTo(f.x0Shape...)(gy[0]),
		SumTo(f.x1Shape...)(gy[0]),
	}
}
