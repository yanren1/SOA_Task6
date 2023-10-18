def respond(success, vector=None, boxes=None,reason=None):
    if success:
        if vector == None:

            return {
                "statusCode": 400,
                "body": {"image_vector": None, "error": "Failed to create image vector"}
            }
        elif boxes==None:
            return {
                "statusCode": 200,
                "body": {"image_vector": vector}
            }
        else:
            return {
                "statusCode": 200,
                "body": {"image_vector": vector,"boxes":boxes}
            }

    else:
        return {
            "statusCode": 400,
            "body": {"image_vector": None, "error": reason}
        }