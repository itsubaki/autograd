package function

import (
	"fmt"
	"math"

	"github.com/itsubaki/autograd/variable"
)

func Exp(x *variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &ExpT{}}).Apply(x)
}

type ExpT struct {
	x variable.Data
}

func (f *ExpT) Forward(x variable.Data) variable.Data {
	f.x = x

	y := variable.NewData(len(x))
	for i := 0; i < len(x); i++ {
		y[i] = math.Exp(x[i])
	}

	return y
}

func (f *ExpT) Backward(gy variable.Data) variable.Data {
	grad := variable.NewData(len(f.x))
	for i := 0; i < len(f.x); i++ {
		grad[i] = math.Exp(f.x[i]) * gy[i]
	}

	return grad
}

func (f ExpT) String() string {
	return fmt.Sprintf("%T(%v)", f, f.x)
}
