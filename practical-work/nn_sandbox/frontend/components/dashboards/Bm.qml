import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12

import '..'

Page {
    width: 500
    height: 300
    name: 'BM'
    ColumnLayout {
        width: 500
        height: 300
        Layout.alignment: Qt.AlignHCenter
        Label {
            width: 200
            height: 100
            text: '11月1日作业第3题 BM 玻尔兹曼机'
            Layout.alignment: Qt.AlignHCenter
        }
        Button {
            text: '开始运行'
            onClicked: {
                bmBridge.start_bm()
            }
            Layout.alignment: Qt.AlignHCenter
        }
    }
}