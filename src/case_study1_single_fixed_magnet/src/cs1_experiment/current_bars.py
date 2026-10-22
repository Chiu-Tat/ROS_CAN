"""Lightweight signed current bars; Qt painting does not block on plotting."""
import math

from PyQt5 import QtCore, QtGui, QtWidgets


class CurrentBars(QtWidgets.QWidget):
    def __init__(self, limit_A, parent=None):
        super().__init__(parent)
        self.limit_A = float(limit_A)
        self.samples = [{'value':None,'command':None,'state':'NO DATA'} for _ in range(10)]
        self.setFixedHeight(180)
        self.setMinimumWidth(500)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setToolTip('Filled bars: measured current. Black ticks: commanded current.\n'
                        'Positive current rises above zero; negative falls below zero.\n'
                        'Amber hatching marks stale feedback; missing feedback has no filled bar.')

    def set_samples(self, samples):
        self.samples = [dict(sample) for sample in samples]
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(),QtGui.QColor('#f8fafb'))
        left,right,top,bottom = 44.,self.width()-8.,12.,120.
        middle = (top+bottom)/2
        half = (bottom-top)/2
        font = painter.font();font.setPixelSize(11);painter.setFont(font)
        def y_for(value):
            return middle-max(-1.,min(1.,value/self.limit_A))*half
        for value,text in [(self.limit_A,'+%g A'%self.limit_A),(0.,'0'),(-self.limit_A,'−%g A'%self.limit_A)]:
            y = y_for(value)
            painter.setPen(QtGui.QColor('#8b9ba4' if value==0 else '#dfe6eb'))
            painter.drawLine(QtCore.QPointF(left,y),QtCore.QPointF(right,y))
            painter.setPen(QtGui.QColor('#536875'))
            painter.drawText(QtCore.QRectF(0,y-8,left-6,16),QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter,text)
        cell = (right-left)/10
        for index,sample in enumerate(self.samples):
            x = left+cell*(index+.5)
            width = min(32.,cell*.48)
            track = QtCore.QRectF(x-width/2,top,width,bottom-top)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor('#e9eff3'))
            painter.drawRoundedRect(track,3,3)
            value,command,state = sample['value'],sample['command'],sample['state']
            known = value is not None and math.isfinite(value) and state in ('FRESH','STALE')
            color = '#98a5ad'
            if known:
                color = '#a86d17' if state=='STALE' else '#227a69' if value>=0 else '#326db4'
                if abs(value)>self.limit_A:color = '#b73949'
                painter.setBrush(QtGui.QBrush(QtGui.QColor(color),
                    QtCore.Qt.BDiagPattern if state=='STALE' else QtCore.Qt.SolidPattern))
                y = y_for(value)
                painter.drawRect(QtCore.QRectF(x-width/2,min(y,middle),width,abs(y-middle)))
            if command is not None and math.isfinite(command):
                painter.setPen(QtGui.QPen(QtGui.QColor('#273b48'),2))
                y = y_for(command)
                painter.drawLine(QtCore.QPointF(x-width/2-4,y),QtCore.QPointF(x+width/2+4,y))
            painter.setPen(QtGui.QColor(color))
            painter.drawText(QtCore.QRectF(x-cell/2,125,cell,17),QtCore.Qt.AlignCenter,
                             '%+.3f A'%value if known else '—')
            painter.setPen(QtGui.QColor('#273b48'))
            painter.drawText(QtCore.QRectF(x-cell/2,143,cell,17),QtCore.Qt.AlignCenter,'Coil %d'%(index+1))
            font.setPixelSize(10);painter.setFont(font)
            painter.setPen(QtGui.QColor('#a86d17' if state=='STALE' else '#667d89'))
            painter.drawText(QtCore.QRectF(x-cell/2,161,cell,16),QtCore.Qt.AlignCenter,state)
            font.setPixelSize(11);painter.setFont(font)
        painter.end()
