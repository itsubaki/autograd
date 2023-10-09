package function

import (
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

func Softmax(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &SoftmaxT{}}).ApplyAndFirst(x...)
}

type SoftmaxT struct {
	y *variable.Variable
}

func (f *SoftmaxT) Forward(x ...*variable.Variable) []*variable.Variable {
	max := matrix.BroadcastTo(matrix.Shape(x[0].Data), matrix.MaxAxis1(x[0].Data)) // max(x, axis=1)
	exp := matrix.Exp(matrix.Sub(x[0].Data, max))                                  // exp(x - max)
	sum := matrix.BroadcastTo(matrix.Shape(exp), matrix.SumAxis1(exp))             // sum(y, axis=1)
	y := matrix.Div(exp, sum)                                                      // exp(x - max) / sum

	f.y = variable.NewOf(y...)
	return []*variable.Variable{
		f.y,
	}
}

func (f *SoftmaxT) Backward(gy ...*variable.Variable) []*variable.Variable {
	gyy := Mul(gy[0], f.y)              // gyy = gy * y
	sum := SumTo(len(gyy.Data), 1)(gyy) // sum = sum(gx, axis=1)
	gx := Sub(gyy, Mul(f.y, sum))       // gx = gx - y * sum

	return []*variable.Variable{
		gx,
	}
}
