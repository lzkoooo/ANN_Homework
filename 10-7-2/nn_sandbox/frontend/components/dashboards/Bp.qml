import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12

import '..'

Page {
    name: 'BP算法'
    ColumnLayout {
        GroupBox {
            title: 'Dataset'
            Layout.fillWidth: true
            ComboBox {
                id: datasetCombobox
                anchors.fill: parent
                model: Object.keys(bpBridge.dataset_dict)
                enabled: bpBridge.has_finished
                onActivated: () => {
                    bpBridge.current_dataset_name = currentText
                    dataChart.updateDataset(bpBridge.dataset_dict[datasetCombobox.currentText])
                }
                Component.onCompleted: () => {
                    bpBridge.current_dataset_name = currentText
                    dataChart.updateDataset(bpBridge.dataset_dict[datasetCombobox.currentText])
                }
            }
        }
        GroupBox {
            title: 'Activation Function'
            Layout.fillWidth: true
            ComboBox {
                id: activationCombobox
                anchors.fill: parent
                model: ['sigmoid', 'bipolar sigmoid']
                enabled: bpBridge.has_finished
                onActivated: bpBridge.activation_function_name = currentText
                Component.onCompleted: bpBridge.activation_function_name = currentText
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
                    enabled: bpBridge.has_finished
                    editable: true
                    value: 10
                    to: 999999
                    onValueChanged: bpBridge.total_epoches = value
                    Component.onCompleted: bpBridge.total_epoches = value
                    Layout.fillWidth: true
                }
                CheckBox {
                    id: mostCorrectRateCheckBox
                    enabled: bpBridge.has_finished
                    text: 'Most Correct Rate'
                    checked: true
                    onCheckedChanged: bpBridge.most_correct_rate_checkbox = checked
                    Component.onCompleted: bpBridge.most_correct_rate_checkbox = checked
                    Layout.alignment: Qt.AlignHCenter
                }
                DoubleSpinBox {
                    enabled: mostCorrectRateCheckBox.checked && bpBridge.has_finished
                    editable: true
                    value: 1.00 * 100
                    onValueChanged: bpBridge.most_correct_rate = value / 100
                    Component.onCompleted: bpBridge.most_correct_rate = value / 100
                    Layout.fillWidth: true
                }
                Label {
                    text: 'Initial Learning Rate'
                    Layout.alignment: Qt.AlignHCenter
                }
                DoubleSpinBox {
                    enabled: bpBridge.has_finished
                    editable: true
                    value: 0.8 * 100
                    onValueChanged: bpBridge.initial_learning_rate = value / 100
                    Component.onCompleted: bpBridge.initial_learning_rate = value / 100
                    Layout.fillWidth: true
                }
                // Label {
                //     text: 'Search Iteration Constant'
                //     Layout.alignment: Qt.AlignHCenter
                // }
                // SpinBox {
                //     enabled: mlpBridge.has_finished
                //     editable: true
                //     value: 10000
                //     to: 999999
                //     onValueChanged: mlpBridge.search_iteration_constant = value
                //     Component.onCompleted: mlpBridge.search_iteration_constant = value
                //     Layout.fillWidth: true
                // }
                // Label {
                //     text: 'Momentum Weight'
                //     Layout.alignment: Qt.AlignHCenter
                // }
                // DoubleSpinBox {
                //     enabled: bpBridge.has_finished
                //     editable: true
                //     value: 0.5 * 100
                //     from: 0
                //     to: 99
                //     Layout.fillWidth: true
                // }
                Label {
                    text: 'Test-Train Ratio'
                    Layout.alignment: Qt.AlignHCenter
                }
                DoubleSpinBox {
                    enabled: bpBridge.has_finished
                    editable: true
                    value: 0.3 * 100
                    from: 30
                    to: 90
                    onValueChanged: bpBridge.test_ratio = value / 100
                    Component.onCompleted: bpBridge.test_ratio = value / 100
                    Layout.fillWidth: true
                }
                Label {
                    text: 'UI Refresh Interval'
                    Layout.alignment: Qt.AlignHCenter
                }
                DoubleSpinBox {
                    enabled: bpBridge.has_finished
                    editable: true
                    value: 0 * 100
                    from: 0 * 100
                    to: 5 * 100
                    onValueChanged: bpBridge.ui_refresh_interval = value / 100
                    Component.onCompleted: bpBridge.ui_refresh_interval = value / 100
                    Layout.fillWidth: true
                }
            }
        }
        GroupBox {
            title: 'Network'
            Layout.fillWidth: true
            NetworkSetting {
                enabled: bpBridge.has_finished
                onShapeChanged: bpBridge.network_shape = shape
                Component.onCompleted: bpBridge.network_shape = shape
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
                    startButton.enabled: bpBridge.has_finished
                    startButton.onClicked: () => {
                        bpBridge.start_bp_algorithm()
                        dataChart.clear()
                        dataChart.updateTrainingDataset(bpBridge.training_dataset)
                        dataChart.updateTestingDataset(bpBridge.testing_dataset)
                        rateChart.reset()
                    }
                    stopButton.enabled: !bpBridge.has_finished
                    stopButton.onClicked: bpBridge.stop_bp_algorithm()
                    progressBar.value: (bpBridge.current_iterations + 1) / (totalEpoches.value * bpBridge.training_dataset.length)
                    Layout.columnSpan: 2
                    Layout.fillWidth: true
                }
                Label {
                    text: '激活函数'
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    text: bpBridge.activation_function_name
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
                        const epoch = Math.floor(bpBridge.current_iterations / bpBridge.training_dataset.length) + 1
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
                    text: bpBridge.current_iterations + 1
                    horizontalAlignment: Text.AlignHCenter
                    onTextChanged: () => {
                        rateChart.bestCorrectRate.append(
                            bpBridge.current_iterations + 1,
                            bpBridge.best_correct_rate
                        )
                        rateChart.trainingCorrectRate.append(
                            bpBridge.current_iterations + 1,
                            bpBridge.current_correct_rate
                        )
                        rateChart.testingCorrectRate.append(
                            bpBridge.current_iterations + 1,
                            bpBridge.test_correct_rate
                        )
                    }
                    Layout.fillWidth: true
                }
                Label {
                    text: '学习率'
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    text: bpBridge.current_learning_rate.toFixed(toFixedValue)
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
                Label {
                    text: '训练集最佳正确率'
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    text: bpBridge.best_correct_rate.toFixed(toFixedValue)
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
                Label {
                    text: '训练集正确率'
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    text: bpBridge.current_correct_rate.toFixed(toFixedValue)
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
                Label {
                    text: '测试集正确率'
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    text: bpBridge.test_correct_rate.toFixed(toFixedValue)
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
            }
        }
    }
    ColumnLayout {
        DataChart {
            id: dataChart
            width: 700
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
        RateChart {
            id: rateChart
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }
}