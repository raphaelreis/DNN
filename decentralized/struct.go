package decentralized

import (
	"go.dedis.ch/onet/v3"
)

/*
Struct holds the messages that will be sent around in the protocol. You have
to define each message twice: once the actual message, and a second time
with the `*onet.TreeNode` embedded. The latter is used in the handler-function
so that it can find out who sent the message.
*/

// Name can be used from other packages to refer to this protocol.
const Name = "Template"

// Structs
//______________________________________________________________________________________________________________________
type newIterationAnnouncementStruct struct {
	*onet.TreeNode
	NewIterationMessage
}

type ChildUpdatedLocalWeightsMessageStruct struct {
	*onet.TreeNode
	ChildUpdatedLocalWeightsMessage
}

type DataAccessStruct struct {
	*onet.TreeNode
	DataAccessMessage
}

// Messages
//______________________________________________________________________________________________________________________

// NewIterationMessage
type NewIterationMessage struct {
	IterationNumber int
	GlobalWeights   []byte
}

// ChildUpdatedLocalWeightsMessage contains one node's weights update.
type ChildUpdatedLocalWeightsMessage struct {
	ChildLocalWeights []byte
	FCounter          int
}

// DataAccessMessage contains the access to the data memory locality
type DataAccessMessage struct {
	TrainData []byte
}
