from krita import DockWidgetFactory, DockWidgetFactoryBase, Krita
from .docker import KritaiDocker

Krita.instance().addDockWidgetFactory(
    DockWidgetFactory("kritai", DockWidgetFactoryBase.DockRight, KritaiDocker)
)
