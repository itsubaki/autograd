package layer

import "github.com/itsubaki/autograd/variable"

type Parameter = *variable.Variable

type Parameters map[string]Parameter

func (p Parameters) Add(name string, param Parameter) {
	param.Name = name
	p[name] = param
}

func (p Parameters) Delete(name string) {
	delete(p, name)
}

func (p Parameters) Params() Parameters {
	return p
}

func (p Parameters) Cleargrads() {
	for k := range p {
		p[k].Cleargrad()
	}
}
