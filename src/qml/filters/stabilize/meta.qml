import QtQuick 2.0
import org.shotcut.qml 1.0

Metadata {
    type: Metadata.Filter
    name: qsTr("Stabilize")
    keywords: qsTr('smooth deshake', 'search keywords for the Stabilize video filter') + ' vid.stab stabilize'
    mlt_service: "vidstab"
    qml: "ui.qml"
    icon: 'icon.webp'
    isClipOnly: true
    allowMultiple: false
    isGpuCompatible: false

    keyframes {
        allowTrim: false
    }

}
