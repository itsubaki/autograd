package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func Softmax(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &SoftmaxT{}}).First(x...)
}

type SoftmaxT struct {
	y *variable.Variable
}

func (f *SoftmaxT) Forward(x ...*variable.Variable) []*variable.Variable {
	max := tensor.Expand(tensor.Max(x[0].Data, 1), 1)
	expy := tensor.Exp(tensor.Sub(x[0].Data, max)) // expy = exp(x - max)
	sumy := tensor.Expand(tensor.Sum(expy, 1), 1)  // sumy = sum(expy, axis=1)

	f.y = variable.NewFrom(tensor.Div(expy, sumy)) // y = expy / sumy
	return []*variable.Variable{
		f.y,
	}
}

func (f *SoftmaxT) Backward(gy ...*variable.Variable) []*variable.Variable {
	gyy := Mul(gy[0], f.y) // gyy = gy * y
	N := gyy.Shape()[0]
	sum := SumTo(N, 1)(gyy) // sum = sum(gx, axis=1)

	return []*variable.Variable{
		Sub(gyy, Mul(f.y, sum)), // gyy - y * sum
	}
}
