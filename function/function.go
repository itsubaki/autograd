package function

import "github.com/itsubaki/autograd/variable"

type Forwarder interface {
	Forward(x variable.Data) variable.Data
	Backward(gy variable.Data) variable.Data
	String() string
}

type Function struct {
	X, Y    variable.Data
	in, out *variable.Variable
	Forwarder
}

func (f *Function) Input() *variable.Variable {
	return f.in
}

func (f *Function) Output() *variable.Variable {
	return f.out
}

func (f *Function) Apply(x *variable.Variable) *variable.Variable {
	f.X, f.Y = x.Data, f.Forward(x.Data)
	f.in, f.out = x, &variable.Variable{Data: f.Y, Creator: f}
	return f.out
}

func Data(v ...*variable.Variable) []variable.Data {
	data := make([]variable.Data, len(v))
	for i := range v {
		data[i] = v[i].Data
	}

	return data
}
