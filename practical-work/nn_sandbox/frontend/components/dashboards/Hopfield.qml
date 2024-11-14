import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12

import '..'

Page {
    name: 'Hopfield'
    RowLayout {
        Layout.fillWidth: true
        Layout.fillHeight: true
        ColumnLayout {
            Layout.fillWidth: false
            GroupBox {
                title: 'Dataset'
                Layout.fillWidth: true
                ComboBox {
                    id: datasetCombobox
                    anchors.fill: parent
                    model: Object.keys(hopfieldBridge.dataset_dict)
                    enabled: hopfieldBridge.has_finished
                    onActivated: () => {
                        hopfieldBridge.current_dataset_name = currentText
                        dataChart.updateDataset(hopfieldBridge.dataset_dict[datasetCombobox.currentText])
                    }
                    Component.onCompleted: () => {
                        hopfieldBridge.current_dataset_name = currentText
                        dataChart.updateDataset(hopfieldBridge.dataset_dict[datasetCombobox.currentText])
                    }
                }
            }
            GroupBox {
                title: 'Hopfield Mode'
                Layout.fillWidth: true
                ComboBox {
                    id: hnnModeCombobox
                    anchors.fill: parent
                    model: ['DHNN', 'CHNN']
                    enabled: hopfieldBridge.has_finished
                    onActivated: hopfieldBridge.hnn_mode = currentText
                    Component.onCompleted: hopfieldBridge.hnn_mode = currentText
                }
            }
            GroupBox {
                title: 'Train Mode'
                Layout.fillWidth: true
                ComboBox {
                    id: trainmodeCombobox
                    anchors.fill: parent
                    model: ['异步更新', '同步更新']
                    enabled: hopfieldBridge.has_finished
                    onActivated: hopfieldBridge.train_mode = currentText
                    Component.onCompleted: hopfieldBridge.train_mode = currentText
                }
            }
            GroupBox {
                title: 'Settings'
                Layout.fillWidth: true
                GridLayout {
                    anchors.fill: parent
                    columns: 2
                    Label {
                        text: 'Total Training Epoches'
                        Layout.alignment: Qt.AlignHCenter
                    }
                    SpinBox {
                        id: totalEpoches
                        enabled: hopfieldBridge.has_finished
                        editable: true
                        value: 10
                        to: 999999
                        onValueChanged: hopfieldBridge.total_epoches = value
                        Component.onCompleted: hopfieldBridge.total_epoches = value
                        Layout.fillWidth: true
                    }
                    Label {
                        text: 'Initial Learning Rate'
                        Layout.alignment: Qt.AlignHCenter
                    }
                    DoubleSpinBox {
                        enabled: hopfieldBridge.has_finished
                        editable: true
                        value: 0.8 * 100
                        onValueChanged: hopfieldBridge.initial_learning_rate = value / 100
                        Component.onCompleted: hopfieldBridge.initial_learning_rate = value / 100
                        Layout.fillWidth: true
                    }
                    Label {
                        text: 'UI Refresh Interval'
                        Layout.alignment: Qt.AlignHCenter
                    }
                    DoubleSpinBox {
                        enabled: hopfieldBridge.has_finished
                        editable: true
                        value: 0 * 100
                        from: 0 * 100
                        to: 5 * 100
                        onValueChanged: hopfieldBridge.ui_refresh_interval = value / 100
                        Component.onCompleted: hopfieldBridge.ui_refresh_interval = value / 100
                        Layout.fillWidth: true
                    }
                }
            }
            GroupBox {
                title: 'Information'
                Layout.fillWidth: true
                Layout.fillHeight: true
                GridLayout {
                    anchors.left: parent.left
                    anchors.right: parent.right
                    columns: 2
                    ExecutionControls {
                        startButton.enabled: hopfieldBridge.has_finished
                        startButton.onClicked: () => {
                            hopfieldBridge.start_hopfield_algorithm()
                            dataChart.clear()
                            dataChart.updateTrainingDataset(hopfieldBridge.training_dataset)
                            dataChart.updateTestingDataset(hopfieldBridge.testing_dataset)
                            rateChart.reset()
                        }
                        stopButton.enabled: !hopfieldBridge.has_finished
                        stopButton.onClicked: hopfieldBridge.stop_hopfield_algorithm()
                        progressBar.value: (hopfieldBridge.current_iterations + 1) / (totalEpoches.value * hopfieldBridge.training_dataset.length)
                        Layout.columnSpan: 2
                        Layout.fillWidth: true
                    }
                    Label {
                        text: '激活函数'
                        Layout.alignment: Qt.AlignHCenter
                    }
                    Label {
                        text: hopfieldBridge.activation_function_name
                        horizontalAlignment: Text.AlignHCenter
                        Layout.fillWidth: true
                    }
                    Label {
                        text: '训练轮次'
                        Layout.alignment: Qt.AlignHCenter
                    }
                    Label {
                        text: currentEpoch()
                        horizontalAlignment: Text.AlignHCenter
                        Layout.fillWidth: true

                        function currentEpoch() {
                            const epoch = Math.floor(hopfieldBridge.current_iterations / hopfieldBridge.training_dataset.length) + 1
                            if (isNaN(epoch))
                                return 1
                            return epoch
                        }
                    }
                    Label {
                        text: '训练迭代次数'
                        Layout.alignment: Qt.AlignHCenter
                    }
                    Label {
                        text: hopfieldBridge.current_iterations + 1
                        horizontalAlignment: Text.AlignHCenter
                        Layout.fillWidth: true
                    }
                    Label {
                        text: '学习率'
                        Layout.alignment: Qt.AlignHCenter
                    }
                    Label {
                        text: hopfieldBridge.current_learning_rate.toFixed(toFixedValue)
                        horizontalAlignment: Text.AlignHCenter
                        Layout.fillWidth: true
                    }
                }
            }
        }
        GroupBox {
            title: 'Results'
            Layout.fillWidth: true
            Layout.fillHeight: true
            GridLayout {
                anchors.left: parent.left
                anchors.right: parent.right
                columns: 2
                Label {
                    text: "神经元当前状态"
                    horizontalAlignment: Text.AlignHCenter
                }
                Label {
                    text: hopfieldBridge.states
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
                Label {
                    text: "神经元当前能量"
                    horizontalAlignment: Text.AlignHCenter
                }
                Label {
                    text: hopfieldBridge.energys
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
                Label {
                    text: "吸引子"
                    horizontalAlignment: Text.AlignHCenter
                }
                Label {
                    text: hopfieldBridge.attractors
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
            }
        }
    }
}