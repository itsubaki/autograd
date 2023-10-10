package variable

import (
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/vector"
)

// SubC returns a variable that c - x[0].
func SubC(c float64, x ...*Variable) *Variable {
	return (&Function{Forwarder: &SubT{}}).ApplyAndFirst(Const(c), x[0])
}

// Sub returns a variable that x[0] - x[1].
func Sub(x ...*Variable) *Variable {
	return (&Function{Forwarder: &SubT{}}).ApplyAndFirst(x...)
}

type SubT struct {
	x0Shape, x1Shape []int
}

func (f *SubT) Forward(x ...*Variable) []*Variable {
	f.x0Shape, f.x1Shape = Shape(x[0]), Shape(x[1])

	x0, x1 := matrix.Broadcast(x[0].Data, x[1].Data)
	y := matrix.Sub(x0, x1)
	return []*Variable{
		NewOf(y...),
	}
}

func (f *SubT) Backward(gy ...*Variable) []*Variable {
	gx0 := gy[0]
	gx1 := Neg(gy[0]) // -1.0 * gy

	if vector.Equals(f.x0Shape, f.x1Shape) {
		return []*Variable{
			gx0,
			gx1,
		}
	}

	return []*Variable{
		SumTo(f.x0Shape...)(gx0),
		SumTo(f.x1Shape...)(gx1),
	}
}
