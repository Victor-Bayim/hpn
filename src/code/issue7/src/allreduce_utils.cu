#include "allreduce_utils.h"

void print_allreduce_config(const AllreduceConfig *config) {
    printf("PE %d/%d: IBGDA %s, Algorithm = %s\n",
           config->mype, config->npes,
           config->use_ibgda ? "Enabled" : "Disabled",
           (config->algo == ALLREDUCE_RING ? "Ring" : "Unknown"));
}
