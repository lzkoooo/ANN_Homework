import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12

import '..'

Page {
    width: 500
    height: 300
    name: 'Grad Descent'
    ColumnLayout {
        width: 500
        height: 300
        Layout.alignment: Qt.AlignHCenter
        Label {
            width: 200
            height: 100
            text: '10月11日作业第3题三种梯度下降方法'
            Layout.alignment: Qt.AlignHCenter
        }
        Button {
            text: '开始运行'
            onClicked: {
                gradBridge.start_gradient_descent()
            }
            Layout.alignment: Qt.AlignHCenter
        }
    }
}