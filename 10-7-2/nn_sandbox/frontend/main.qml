import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12

import 'components'
import 'components/dashboards'

ApplicationWindow {
    id: window
    visible: true
    title: '人工神经网络10.7作业-李兆堃-2024216083'
    // XXX: using body.implicitWidth will cause BadValue and BadWindow error in
    // Linux (Kubuntu). Need further research. Currently, I use
    // Component.onCompleted instead as a workaround.

    Pane {
        id: body
        anchors.fill: parent
        NoteBook {
            Bp {}
        }
    }

    Component.onCompleted: () => {
        width = minimumWidth = body.implicitWidth
        height = minimumHeight = 1000
    }
}
