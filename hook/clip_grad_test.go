package hook_test

import (
	"fmt"

	"github.com/itsubaki/autograd/hook"
	"github.com/itsubaki/autograd/variable"
)

func ExampleClipGrad() {
	p := variable.New(1)
	p.Grad = variable.New(1, 2, 3, 4)

	h := hook.ClipGrad(1.0)
	h([]*variable.Variable{p})

	fmt.Println(p.Grad)

	// Output:
	// variable([0.18257415250172812 0.36514830500345624 0.5477224575051843 0.7302966100069125])
}

func ExampleClipGrad_noclip() {
	p := variable.New(1)
	p.Grad = variable.New(0.1, 0.2, 0.3, 0.4)

	h := hook.ClipGrad(1.0)
	h([]*variable.Variable{p})

	fmt.Println(p.Grad)

	// Output:
	// variable([0.1 0.2 0.3 0.4])
}
