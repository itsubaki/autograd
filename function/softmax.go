package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func Softmax(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{
		Forwarder: &SoftmaxT{},
	}).First(x...)
}

type SoftmaxT struct {
	y *variable.Variable
}

func (f *SoftmaxT) Forward(x ...*variable.Variable) []*variable.Variable {
	max1 := tensor.Expand(tensor.Max(x[0].Data, 1), 1) // max1 = max(x, axis=1)
	expy := tensor.Exp(tensor.Sub(x[0].Data, max1))    // expy = exp(x - max1)
	sum1 := tensor.Expand(tensor.Sum(expy, 1), 1)      // sum1 = sum(expy, axis=1)
	div := tensor.Div(expy, sum1)                      // y = expy / sum1

	f.y = variable.From(div)
	return []*variable.Variable{
		f.y,
	}
}

func (f *SoftmaxT) Backward(gy ...*variable.Variable) []*variable.Variable {
	gyy := Mul(gy[0], f.y) // gyy = gy * y
	N := gyy.Shape()[0]
	sum := SumTo(N, 1)(gyy) // sum = sum(gy, axis=1)

	gx := Sub(gyy, Mul(f.y, sum)) // gyy - y * sum
	return []*variable.Variable{
		gx,
	}
}
