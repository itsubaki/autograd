package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Square(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &SquareT{}}).Apply(x...)[0]
}

type SquareT struct {
	x *variable.Variable
}

func (f *SquareT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x = x[0]

	y := vector.Pow(f.x.Data, 2)
	return []*variable.Variable{
		variable.New(y...),
	}
}

func (f *SquareT) Backward(gy ...*variable.Variable) []*variable.Variable {
	c := variable.NewLikeWith(2.0, f.x)
	return []*variable.Variable{
		Mul(c, Mul(f.x, gy[0])),
	}
}
