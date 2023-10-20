package layer

import "github.com/itsubaki/autograd/variable"

type Parameter = *variable.Variable

type Parameters map[string]Parameter

func (p Parameters) Add(name string, param Parameter) {
	param.Name = name
	p[name] = param
}

func (p Parameters) Params() []Parameter {
	params := make([]Parameter, 0)
	for k := range p {
		params = append(params, p[k])
	}

	return params
}

func (p Parameters) Cleargrads() {
	for k := range p {
		p[k].Cleargrad()
	}
}
