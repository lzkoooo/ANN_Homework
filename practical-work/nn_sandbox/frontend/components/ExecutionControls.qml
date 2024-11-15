import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12

Pane {
    property alias startButton: startButton
    property alias stopButton: stopButton
    property alias applyButton1: applyButton1
    property alias applyButton2: applyButton2
    property alias applyButton3: applyButton3
    property alias applyButton4: applyButton4
    property alias applyButton5: applyButton5
    property alias applyButton6: applyButton6
    property alias progressBar: progressBar
    
    ColumnLayout {
        anchors.fill: parent
        RowLayout {
            RoundButton {
                id: startButton
                icon.source: '../../assets/images/baseline-play_arrow-24px.svg'
                radius: 0
                ToolTip.visible: hovered
                ToolTip.text: 'Start Training'
            }
            RoundButton {
                id: stopButton
                icon.source: '../../assets/images/baseline-stop-24px.svg'
                radius: 0
                ToolTip.visible: hovered
                ToolTip.text: 'Stop Training'
            }
            Button {
                id: applyButton1
                text: 'Apply'
                visible: false
                enabled: false
                ToolTip.visible: hovered
                ToolTip.text: '需要训练'
            }
            Button {
                id: applyButton2
                text: 'Apply'
                visible: false
                enabled: false
                ToolTip.visible: hovered
                ToolTip.text: '需要训练'
            }
            Button {
                id: applyButton3
                text: 'Apply'
                visible: false
                enabled: false
                ToolTip.visible: hovered
                ToolTip.text: '不需要训练'
            }
            Button {
                id: applyButton4
                text: 'Apply'
                visible: false
                enabled: false
                ToolTip.visible: hovered
                ToolTip.text: '不需要训练'
            }
            Button {
                id: applyButton5
                text: 'Apply'
                visible: false
                enabled: false
                ToolTip.visible: hovered
                ToolTip.text: '不需要训练'
            }
            Button {
                id: applyButton6
                text: 'Apply'
                visible: false
                enabled: false
                ToolTip.visible: hovered
                ToolTip.text: '不需要训练'
            }
        }
        ProgressBar {
            id: progressBar
            Layout.fillWidth: true
        }
    }
}