package layer

type Model struct {
	Layers []*Layer
}

func (m *Model) Cleargrads() {
	for _, l := range m.Layers {
		l.Cleargrads()
	}
}
