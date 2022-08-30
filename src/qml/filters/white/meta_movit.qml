import QtQuick 2.0
import org.shotcut.qml 1.0

Metadata {
    type: Metadata.Filter
    name: qsTr("White Balance")
    keywords: qsTr('color correct light temperature neutral', 'search keywords for the White Balance video filter') + ' white balance'
    mlt_service: "movit.white_balance"
    needsGPU: true
    qml: "ui.qml"
    icon: 'icon.webp'
    isFavorite: true
}
