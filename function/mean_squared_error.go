package function

import (
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

func MeanSquaredError(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &MeanSquaredErrorT{}}).ApplyAndFirst(x...)
}

type MeanSquaredErrorT struct {
	x0, x1 *variable.Variable
}

func (f *MeanSquaredErrorT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x0, f.x1 = x[0], x[1]

	diff := matrix.Sub(x[0].Data, x[1].Data)
	N := len(diff)

	y := (1.0 / float64(N)) * matrix.Sum(matrix.Mul(diff, diff)) // (1/N) * sum((x0 - x1)^2)
	return []*variable.Variable{
		variable.New(y),
	}
}

func (f *MeanSquaredErrorT) Backward(gy ...*variable.Variable) []*variable.Variable {
	diff := Sub(f.x0, f.x1)
	N, gyb := len(diff.Data), BroadcastTo(variable.Shape(diff)...)(gy[0])

	gx0 := MulC(2.0/float64(N), Mul(gyb, diff)) // gy * (x0 - x1) * 2/N
	gx1 := Neg(gx0)                             // -gx0
	return []*variable.Variable{
		gx0,
		gx1,
	}
}
