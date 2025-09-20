package layer

import (
	"iter"
	"slices"

	"github.com/itsubaki/autograd/variable"
)

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

func (p Parameters) Seq2() iter.Seq2[string, Parameter] {
	keys := make([]string, 0, len(p))
	for k := range p {
		keys = append(keys, k)
	}
	slices.Sort(keys)

	return func(yield func(string, Parameter) bool) {
		for _, k := range keys {
			if !yield(k, p[k]) {
				return
			}
		}
	}
}
