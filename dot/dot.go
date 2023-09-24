package dot

import (
	"fmt"
	"strings"
	"unsafe"

	"github.com/itsubaki/autograd/variable"
)

func Var(v *variable.Variable) string {
	format := "%d [label=\"%v\", color=orange, style=filled]"
	return fmt.Sprintf(format, unsafe.Pointer(v), v.Name)
}

func Func(f variable.Function) []string {
	format := "%d [label=\"%s\", color=lightblue, style=filled, shape=box]"

	str := fmt.Sprintf("%s", f)
	begin, end := strings.Index(str, "."), strings.Index(str, "[")
	out := []string{fmt.Sprintf(format, unsafe.Pointer(&f), str[begin+1:end])}

	for _, v := range f.Input() {
		out = append(out, fmt.Sprintf("%d -> %d", unsafe.Pointer(v), unsafe.Pointer(&f)))
	}

	for _, v := range f.Output() {
		out = append(out, fmt.Sprintf("%d -> %d", unsafe.Pointer(&f), unsafe.Pointer(v)))
	}

	return out
}

func Graph(v *variable.Variable) []string {
	seen := make(map[variable.Function]bool)
	fs := addFunc(make([]variable.Function, 0), v.Creator, seen)

	out := append([]string{"digraph g {"}, Var(v))
	for {
		if len(fs) == 0 {
			break
		}

		// pop
		f := fs[len(fs)-1]
		fs = fs[:len(fs)-1]
		out = append(out, Func(f)...)

		x := f.Input()
		for i := range x {
			out = append(out, Var(x[i]))

			if x[i].Creator != nil {
				fs = addFunc(fs, x[i].Creator, seen)
			}
		}
	}

	out = append(out, "}")
	return out
}

func addFunc(fs []variable.Function, f variable.Function, seen map[variable.Function]bool) []variable.Function {
	if _, ok := seen[f]; ok {
		return fs
	}

	seen[f] = true
	fs = append(fs, f)
	return fs
}
