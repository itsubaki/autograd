package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleVariable() {
	v := variable.New(1, 2, 3, 4)
	fmt.Println(v)

	// Output:
	// variable[1 2 3 4]
}

func ExampleVariable_Name() {
	v := variable.New(1, 2, 3, 4)
	v.Name = "v"
	fmt.Println(v)

	// Output:
	// v[1 2 3 4]
}
